# Main script for EE32009 Coursework: spike detection and classification
#
# Pipeline overview
#   1. Load data and remove baseline drift
#   2. Estimate noise levels of D1-D6
#   3. Split D1 into train / validation sets (by time)
#   4. Build training windows (clean D1) and validation windows (D1 + noise per level)
#   5. Normalisation constants and the noise augmentation used during training
#   6. Build data loaders
#   7. Train (or load) the classifier and the detector
#   8. Validate the classifier
#   9. Sweep detector thresholds on validation data (optional)
#  10. Predict D2-D6 and export the .mat submission files

# ----------------------------------------------------------------------
# 0. Imports and settings
# ----------------------------------------------------------------------
from pathlib import Path
import copy
import numpy as np
import scipy.io as spio
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

import Functions
import Models
from Config import DATA_DIR, DEVICE, DET_PRE, DET_POST, CLS_PRE, CLS_POST

RETRAIN = False       # True: train and save models. False: load saved models.
RUN_SWEEP = False    # True: print the threshold sweep (slow, only needed when tuning)
RUN_COMPARE = True

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
LOWPASS_HZ = 4000   # must match the cutoff your saved classifier was built with

# Detection settings per dataset: {dataset number: (threshold, min spike distance)}
settings = {2: (0.7, 15), 3: (0.9, 15), 4: (0.9, 25), 5: (0.8, 25), 6: (0.8, 25)}

rng = np.random.default_rng(0)
torch.manual_seed(0)

# ----------------------------------------------------------------------
# 1. Load data and remove baseline drift
# ----------------------------------------------------------------------
# ds = [D1, D2, D3, D4, D5, D6]; D1 is labelled (Index1, Class1)
ds, Index1, Class1 = Functions.load_data(DATA_DIR)
ds = [Functions.remove_baseline(s) for s in ds]   # same cleaning for train and test data
d1 = ds[0]

# ----------------------------------------------------------------------
# 2. Estimate the noise level of every dataset
# ----------------------------------------------------------------------
base = Functions.noise_std(d1)                        # D1's own noise level
for n, sig in enumerate(ds, 1):
    print(f'D{n} noise std: {Functions.noise_std(sig):.4f}')
targets = [Functions.noise_std(sig) for sig in ds[1:]]  # measured noise of D2..D6
levels = [base] + targets                             # noise levels of D1..D6

# ----------------------------------------------------------------------
# 3. Split D1 into training and validation parts (by time)
# ----------------------------------------------------------------------
M = 30                                    # margin so jittered windows stay inside the signal
split = int(0.8 * len(d1))                # first 80% of the recording = training

# Keep only spikes whose windows fit inside the recording (labels filtered together)
ok = (Index1 > CLS_PRE + M) & (Index1 < len(d1) - CLS_POST - M)
Index1, Class1 = Index1[ok], Class1[ok]

# Gap around the split so training and validation windows never overlap
tr = Index1 < split - CLS_POST - M
va = Index1 >= split + CLS_PRE + M
tr_Index1, tr_Class1 = Index1[tr], Class1[tr]
va_Index1, va_Class1 = Index1[va], Class1[va]

# Background (no-spike) positions, split the same way. Must be >100 samples from any spike.
cand = rng.integers(CLS_PRE + M, len(d1) - CLS_POST - M, size=30000)
dist = np.abs(cand[:, None] - Index1[None, :]).min(axis=1)
cand = cand[dist > 100]
neg_tr = cand[cand < split - CLS_POST - M][:3 * len(tr_Index1)]
neg_va = cand[cand >= split + CLS_PRE + M][:3 * len(va_Index1)]

# Hard negatives: positions 15-60 samples from a spike (teaches the detector exact timing)
hn_tr = Functions.hard_negatives(tr_Index1, DET_PRE + M, split - DET_POST - M, Index1, rng)
hn_va = Functions.hard_negatives(va_Index1, split + DET_PRE + M, len(d1) - DET_POST - M, Index1, rng)

# ----------------------------------------------------------------------
# 4. Build training and validation windows
# ----------------------------------------------------------------------
# Training data: CLEAN D1 windows. Noise is added fresh every batch during training (section 5).
# Classifier: spike windows only, with timing jitter
Xc = Functions.windows(d1, tr_Index1, CLS_PRE, CLS_POST, jitter=10, rng=rng)
yc = tr_Class1.astype(int) - 1            # labels 1-5 -> 0-4

# Detector: spikes (label 1), background (0) and hard negatives (0)
c3 = tr_Index1[tr_Class1 == 3]
c3_rep = np.repeat(c3, 3)                       # 3 extra jittered copies of every Class 3 spike

Xd = np.concatenate([Functions.windows(d1, tr_Index1, DET_PRE, DET_POST, jitter=10, rng=rng),
                     Functions.windows(d1, c3_rep, DET_PRE, DET_POST, jitter=10, rng=rng),
                     Functions.windows(d1, neg_tr, DET_PRE, DET_POST),
                     Functions.windows(d1, hn_tr, DET_PRE, DET_POST)])
yd = np.concatenate([np.ones(len(tr_Index1)),
                     np.ones(len(c3_rep)),
                     np.zeros(len(neg_tr)),
                     np.zeros(len(hn_tr))])

# Validation data: FIXED noisy copies of the held-out part of D1, one per noise level D1..D6
det_val = []      # detector validation sets, one (X, y) per level
val_sets = []     # classifier validation sets, one dict per level
for level in levels:
    sig = Functions.noisy_version(d1, level, base, rng)

    Xv = np.concatenate([Functions.windows(sig, va_Index1, DET_PRE, DET_POST),
                         Functions.windows(sig, neg_va, DET_PRE, DET_POST),
                         Functions.windows(sig, hn_va, DET_PRE, DET_POST)])
    yv = np.concatenate([np.ones(len(va_Index1)), np.zeros(len(neg_va)), np.zeros(len(hn_va))])
    det_val.append((Xv, yv))

    val_sets.append({'level': level,
                     'X': Functions.windows(sig, va_Index1, CLS_PRE, CLS_POST),
                     'y': va_Class1 - 1})

Xc_va = np.concatenate([v['X'] for v in val_sets])
yc_va = np.concatenate([v['y'] for v in val_sets])

# ----------------------------------------------------------------------
# 5. Normalisation constants and noise augmentation
# ----------------------------------------------------------------------
# Every window is normalised as (x - MU) / SIGMA, in training and at prediction time
MU, SIGMA = float(np.median(d1)), float(d1.std())

# Extra noise std (in normalised units) needed to bring D1 up to each dataset's noise level
levels_extra = torch.tensor(
    [np.sqrt(max(l**2 - base**2, 0)) for l in levels],
    dtype=torch.float32, device=DEVICE) / SIGMA

# Applied to each training batch: random level from D1..D6, fresh Gaussian noise every time
# Sampling weights for noise levels D1..D6 (D1 = clean, a small share keeps it in training)
level_weights = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], device=DEVICE)

augment = lambda xb: Functions.add_noise(xb, levels_extra, level_weights)


# ----------------------------------------------------------------------
# 6. Data loaders
# ----------------------------------------------------------------------
train_dl = Functions.make_loader(Xc, yc, shuffle=True, mu=MU, sigma=SIGMA)
val_dl = Functions.make_loader(Xc_va, yc_va, shuffle=False, mu=MU, sigma=SIGMA)

det_dl = Functions.make_loader(Xd, yd, shuffle=True, mu=MU, sigma=SIGMA)
det_val_dl = Functions.make_loader(np.concatenate([x for x, _ in det_val]),
                                   np.concatenate([y for _, y in det_val]),
                                   shuffle=False, mu=MU, sigma=SIGMA)

# ----------------------------------------------------------------------
# 7. Train or load the two models
#    clf: which of the 5 neuron classes is this spike?   (5 outputs)
#    det: is there a spike starting here?                (2 outputs)
# ----------------------------------------------------------------------
clf = Models.SpikeCNN(n_out=5)
det = Models.SpikeCNN(n_out=2)

if RETRAIN:
    clf = Functions.train_model(clf, train_dl, val_dl, nn.CrossEntropyLoss(), epochs=90, patience=12, augment=augment)
    det = Functions.train_model(det, det_dl, det_val_dl, nn.CrossEntropyLoss(), epochs=70, patience=12, augment=augment)
    torch.save(clf.state_dict(), MODEL_DIR / 'clf.pt')
    torch.save(det.state_dict(), MODEL_DIR / 'det.pt')
else:
    clf.load_state_dict(torch.load(MODEL_DIR / 'clf.pt', map_location=DEVICE))
    det.load_state_dict(torch.load(MODEL_DIR / 'det.pt', map_location=DEVICE))

clf.to(DEVICE).eval()
det.to(DEVICE).eval()

# ----------------------------------------------------------------------
# 8. Validate the classifier at each noise level (D1..D6)
# ----------------------------------------------------------------------
for v in val_sets:
    X = torch.tensor(((v['X'] - MU) / SIGMA)[:, None, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = clf(X).argmax(1).cpu().numpy()
    acc = (pred == v['y']).mean()

    # Same check with jittered windows (closer to detections that are slightly off)
    sig = Functions.noisy_version(d1, v['level'], base, rng)
    Xj = Functions.windows(sig, va_Index1, CLS_PRE, CLS_POST, jitter=10, rng=rng)
    acc_j = Functions.accuracy(clf, Xj, v['y'], MU, SIGMA)
    templates = Functions.class_templates(d1, tr_Index1, tr_Class1, CLS_PRE, CLS_POST)
    acc_t = (Functions.template_classify(v['X'], templates) == v['y']).mean()
    acc_tj = (Functions.template_classify(Xj, templates) == v['y']).mean()

    print(f"   template classifier: aligned {acc_t:.3f} | jittered {acc_tj:.3f}")
    print(f"noise std {v['level']:.3f} | aligned acc {acc:.3f} | jittered acc {acc_j:.3f}")
    print(confusion_matrix(v['y'], pred))

def detection_report(Index, cls, level):
    pairs = Functions.match_pairs(Index, va_Index1)
    pi = np.array([p for p, _ in pairs])
    tj = np.array([t for _, t in pairs])
    off = Index[pi] - va_Index1[tj]
    true_c, pred_c = va_Class1[tj] - 1, cls[pi]
    print(f'noise {level:.3f}: matched {len(pi)} of {len(va_Index1)}')
    for k in range(5):
        m = true_c == k
        print(f'  class {k + 1}: n {m.sum():3d} | label acc {np.mean(pred_c[m] == k):.2f} | '
              f'offset median {np.median(off[m]):+.0f}, 90% within +-{np.percentile(np.abs(off[m]), 90):.0f}')
    print(confusion_matrix(true_c, pred_c))

start, stop = split + DET_PRE + M, len(d1) - DET_POST - M
for n in (4, 5, 6):                                   # D4, D5, D6 noise levels
    level = levels[n - 1]
    sig = Functions.noisy_version(d1, level, base, rng)
    probs = Functions.detect_probs(det, sig, start, stop, mu=MU, sigma=SIGMA)
    peaks, _ = find_peaks(probs, height=0.8, distance=25)
    Index = peaks + start
    Index = Index[Index < len(sig) - CLS_POST]
    cls = Functions.classify(sig, Index, clf, MU, SIGMA)
    templates = Functions.class_templates(d1, tr_Index1, tr_Class1, CLS_PRE, CLS_POST)
    cls_t = Functions.template_classify_shift(sig, Index, templates)
    pairs = Functions.match_pairs(Index, va_Index1)
    pi = np.array([p for p, _ in pairs])
    tj = np.array([t for _, t in pairs])
    true_c = va_Class1[tj] - 1
    print(f'noise {level:.3f} | label acc on found spikes: CNN {np.mean(cls[pi] == true_c):.3f} '
        f'| template+shift {np.mean(cls_t[pi] == true_c):.3f}')
    detection_report(Index, cls, level)
# ----------------------------------------------------------------------
# 9. Detector threshold sweep on validation data (optional)
#    Use it to choose the (threshold, distance) pair for each dataset in `settings`.
# ----------------------------------------------------------------------
WEIGHTS = {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30}    # F_Final weights
MF_Z_LIST = (None, 3.5, 4.0, 4.5)                          # None = CNN only (your current result)

if RUN_SWEEP:
    start, stop = split + DET_PRE + M, len(d1) - DET_POST - M

    # Class 3 template from clean D1 training spikes
    tmpl3 = Functions.class_template(d1, tr_Index1, tr_Class1, k=3)

    best_cnn, best_mf = {}, {}        # dataset number -> (F_dataset, thr, dist, mf_z)
    for n, level in enumerate(levels, 1):                  # n = 1 is D1, n = 2 is D2, ...
        sig = Functions.noisy_version(d1, level, base, rng)
        probs = Functions.detect_probs(det, sig, start, stop, mu=MU, sigma=SIGMA)
        score = Functions.matched_filter_score(sig, tmpl3)[start:stop]   # validation region only

        for dist_ in (25,):
            for thr in (0.8, 0.9):
                peaks, _ = find_peaks(probs, height=thr, distance=dist_)
                cnn_idx = peaks + start

                for mf_z in MF_Z_LIST:
                    Index = cnn_idx
                    if mf_z is not None:
                        cand, _ = find_peaks(score, height=mf_z, distance=dist_)
                        Index = Functions.add_far_candidates(cnn_idx, cand + start)
                    Index = Index[Index < len(sig) - CLS_POST]   # classifier window must fit

                    pairs = Functions.match_pairs(Index, va_Index1)
                    found = np.zeros(len(va_Index1), bool)
                    for _, j in pairs:
                        found[j] = True
                    cls = Functions.classify(sig, Index, clf, MU, SIGMA)
                    f_id, f_cl, f_ds = Functions.dataset_score(Index, cls, va_Index1, va_Class1 - 1)
                    rec = ' '.join(f'{found[va_Class1 == k].mean():.2f}' for k in range(1, 6))
                    print(f'D{n} noise {level:.3f} | dist {dist_} | thr {thr} | mf_z {mf_z} | '
                          f'F_ident {f_id:.3f} F_class {f_cl:.3f} F_dataset {f_ds:.3f} | class recall {rec}')

                    table = best_cnn if mf_z is None else best_mf
                    if n >= 2 and f_ds > table.get(n, (-1,))[0]:
                        table[n] = (f_ds, thr, dist_, mf_z)

    print('\nBest per dataset (CNN only | with matched filter):')
    for n in sorted(best_cnn):
        a, b = best_cnn[n], best_mf[n]
        print(f'D{n}: {a[0]:.3f} (thr {a[1]}) | {b[0]:.3f} (thr {b[1]}, mf_z {b[3]})')
    for name, table in (('CNN only', best_cnn), ('with matched filter', best_mf)):
        print(f'Estimated F_Final, {name}: {sum(WEIGHTS[n] * table[n][0] for n in WEIGHTS):.3f}')
# ----------------------------------------------------------------------
# 10. Predict D2-D6 and export Index/Class files for the automarker
# ----------------------------------------------------------------------
out_dir = Path('submission')
out_dir.mkdir(exist_ok=True)

print(f'D1 reference: {len(Index1)} spikes')
for n in range(2, 7):
    thr, dist_ = settings[n]
    Index, Class = Functions.predict_dataset(ds[n - 1], thr, dist_, det, clf, MU, SIGMA)
    print(f'D{n}: {len(Index)} spikes, class counts {np.bincount(Class, minlength=6)[1:]}')
    # Only Index and Class are saved (no raw signal d)
    spio.savemat(out_dir / f'D{n}.mat', {'Index': Index.astype(np.int64),
                                         'Class': Class.astype(np.int64)})

# Check one saved file
m = spio.loadmat(out_dir / 'D2.mat', squeeze_me=True)
print(m.keys())      # should show Index and Class (plus __header__ entries)

#test models
# ----------------------------------------------------------------------
# 11. Compare classifier variants on identical detections (optional)
#
# Paste this block at the END of Main.py (after section 10) and add
#     RUN_COMPARE = True
# next to RUN_SWEEP at the top. Run with RETRAIN = False so the saved detector
# is loaded. It uses variables already defined in Main.py (det, clf, d1, base,
# rng, levels, level_weights, augment, val_dl, MU, SIGMA, split, M, ...).
#
# Every classifier is scored on exactly the same detected spikes (same detector,
# same fixed noise draws), so differences come from the classifier alone.
# ----------------------------------------------------------------------
if RUN_COMPARE:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score

    EVAL_LEVELS = (4, 5, 6)        # D4, D5, D6 noise levels
    N_DRAWS = 5                    # noise draws pooled per level
    THR, DIST = 0.8, 25            # detector settings used to make the detections
    SHIFTS = range(-8, 9)          # shifts tried by the template classifier
    LEVEL_WEIGHTS = {4: 0.20, 5: 0.25, 6: 0.30}   # F_Final weights of D4..D6

    # ---------- 1. fixed evaluation set: detections made once ----------
    start, stop = split + DET_PRE + M, len(d1) - DET_POST - M
    eval_sets = {}                 # n -> (windows, shifted windows, true labels)
    for n in EVAL_LEVELS:
        r = np.random.default_rng(1000 + n)
        X_all, Xs_all, y_all = [], [], []
        for _ in range(N_DRAWS):
            sig = Functions.noisy_version(d1, levels[n - 1], base, r)
            probs = Functions.detect_probs(det, sig, start, stop, mu=MU, sigma=SIGMA)
            peaks, _ = find_peaks(probs, height=THR, distance=DIST)
            Index = peaks + start
            Index = Index[Index < len(sig) - CLS_POST]
            pairs = Functions.match_pairs(Index, va_Index1)       # keep detections that hit a real spike
            pi = np.array([p for p, _ in pairs])
            tj = np.array([t for _, t in pairs])
            pos = Index[pi]
            X_all.append(Functions.windows(sig, pos, CLS_PRE, CLS_POST))
            Xs_all.append(np.stack(
                [Functions.windows(sig, np.clip(pos + s, CLS_PRE, len(sig) - CLS_POST),
                                   CLS_PRE, CLS_POST) for s in SHIFTS], axis=1))
            y_all.append(va_Class1[tj] - 1)
        eval_sets[n] = (np.concatenate(X_all), np.concatenate(Xs_all), np.concatenate(y_all))
        print(f'D{n} level: {len(eval_sets[n][2])} matched spikes pooled over {N_DRAWS} draws')

    # ---------- 2. helpers ----------
    def cnn_probs_w(model, X):
        """Softmax class probabilities (n, 5) of a CNN on raw windows X."""
        Xt = torch.tensor(((X - MU) / SIGMA)[:, None, :], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.softmax(model(Xt[i:i + 4096].to(DEVICE)), 1).cpu()
                              for i in range(0, len(Xt), 4096)]).numpy()

    def train_cnn(jitter, epochs, seed):
        """New classifier CNN trained with the given position jitter (samples)."""
        torch.manual_seed(seed)
        Xj = Functions.windows(d1, tr_Index1, CLS_PRE, CLS_POST, jitter=jitter, rng=rng)
        dl = Functions.make_loader(Xj, tr_Class1.astype(int) - 1, shuffle=True, mu=MU, sigma=SIGMA)
        m = copy.deepcopy(clf)
        for layer in m.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()      # fresh random weights
        m = m.to(DEVICE)
        return Functions.train_model(m, dl, val_dl, nn.CrossEntropyLoss(),
                                     epochs=epochs, patience=12, augment=augment)

    lvl_p = level_weights.cpu().numpy()
    lvl_p = lvl_p / lvl_p.sum()
    lvl_extra = np.sqrt(np.maximum(np.array(levels) ** 2 - base ** 2, 0))

    def noisy_copies(X, copies):
        """`copies` noisy versions of every window, noise level drawn like in CNN training."""
        Xr = np.repeat(X, copies, axis=0)
        idx = rng.choice(len(levels), size=len(Xr), p=lvl_p)
        return Xr + rng.normal(0, 1, Xr.shape) * lvl_extra[idx][:, None]

    def fit_sklearn(model, jitter=5, copies=20):
        Xj = Functions.windows(d1, tr_Index1, CLS_PRE, CLS_POST, jitter=jitter, rng=rng)
        Xn = (noisy_copies(Xj, copies) - MU) / SIGMA
        yn = np.repeat(tr_Class1.astype(int) - 1, copies)
        return model.fit(Xn, yn)

    templates = Functions.class_templates(d1, tr_Index1, tr_Class1, CLS_PRE, CLS_POST)

    def template_probs_shift(Xs, n):
        """Best-shift squared distance to each class template, turned into probabilities
        assuming white Gaussian noise of dataset n's level."""
        d = np.concatenate([
            ((Xs[i:i + 400][:, :, None, :] - templates[None, None, :, :]) ** 2).sum(axis=3).min(axis=1)
            for i in range(0, len(Xs), 400)])
        logit = -d / (2 * levels[n - 1] ** 2)
        p = np.exp(logit - logit.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    # ---------- 3. train the candidates ----------
    print('\nTraining candidates (this can take a while)...')
    cnn_j5 = train_cnn(jitter=5, epochs=80, seed=1)
    cnn_j3 = train_cnn(jitter=3, epochs=80, seed=2)
    cnn_j5b = train_cnn(jitter=5, epochs=80, seed=3)
    cnn_j5c = train_cnn(jitter=5, epochs=80, seed=4)
    lda = fit_sklearn(LDA())
    pca_lda = fit_sklearn(make_pipeline(PCA(n_components=20), LDA()))

    # ---------- 4. score every candidate on the same detections ----------
    candidates = {
        'CNN current (jitter 10, 40 ep)': lambda X, Xs, n: cnn_probs_w(clf, X),
        'CNN jitter 5, 80 ep':            lambda X, Xs, n: cnn_probs_w(cnn_j5, X),
        'CNN jitter 3, 80 ep':            lambda X, Xs, n: cnn_probs_w(cnn_j3, X),
        'CNN ensemble x3 (jitter 5)':     lambda X, Xs, n: (cnn_probs_w(cnn_j5, X) + cnn_probs_w(cnn_j5b, X)
                                                            + cnn_probs_w(cnn_j5c, X)) / 3,
        'LDA':                            lambda X, Xs, n: lda.predict_proba((X - MU) / SIGMA),
        'PCA(20) + LDA':                  lambda X, Xs, n: pca_lda.predict_proba((X - MU) / SIGMA),
        'Template + shift':               lambda X, Xs, n: template_probs_shift(Xs, n),
        'Blend: CNN current + template':  lambda X, Xs, n: 0.5 * cnn_probs_w(clf, X) + 0.5 * template_probs_shift(Xs, n),
        'Blend: CNN ens. + template':     lambda X, Xs, n: 0.5 * ((cnn_probs_w(cnn_j5, X) + cnn_probs_w(cnn_j5b, X)
                                                                   + cnn_probs_w(cnn_j5c, X)) / 3)
                                                           + 0.5 * template_probs_shift(Xs, n),
    }

    print('\nLabel accuracy / macro-F1 on found spikes (same detections for every row)')
    print(f"{'classifier':34s}" + ''.join(f'|   D{n} acc  macro-F1 ' for n in EVAL_LEVELS) + '| weighted macro-F1')
    for name, fn in candidates.items():
        row, wsum = '', 0.0
        for n in EVAL_LEVELS:
            X, Xs, y = eval_sets[n]
            pred = fn(X, Xs, n).argmax(1)
            acc = np.mean(pred == y)
            f1 = f1_score(y, pred, average='macro', labels=range(5), zero_division=0)
            se = np.sqrt(acc * (1 - acc) / (len(y) / N_DRAWS))   # draws overlap, so use one draw's size
            row += f'| {acc:.3f}+-{se:.3f}  {f1:.3f}  '
            wsum += LEVEL_WEIGHTS[n] * f1
        print(f'{name:34s}{row}| {wsum / sum(LEVEL_WEIGHTS.values()):.3f}')
