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
RUN_SWEEP = True    # True: print the threshold sweep (slow, only needed when tuning)

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Detection settings per dataset: {dataset number: (threshold, min spike distance)}
settings = {2: (0.9, 15), 3: (0.85, 15), 4: (0.93, 25), 5: (0.9, 25), 6: (0.85, 25)}

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
Xd = np.concatenate([Functions.windows(d1, tr_Index1, DET_PRE, DET_POST, jitter=10, rng=rng),
                     Functions.windows(d1, neg_tr, DET_PRE, DET_POST),
                     Functions.windows(d1, hn_tr, DET_PRE, DET_POST)])
yd = np.concatenate([np.ones(len(tr_Index1)),
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
augment = lambda xb: Functions.add_noise(xb, levels_extra)

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
    clf = Functions.train_model(clf, train_dl, val_dl, nn.CrossEntropyLoss(), epochs=80, patience=12, augment=augment)
    det = Functions.train_model(det, det_dl, det_val_dl, nn.CrossEntropyLoss(), augment=augment)
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

    print(f"noise std {v['level']:.3f} | aligned acc {acc:.3f} | jittered acc {acc_j:.3f}")
    print(confusion_matrix(v['y'], pred))

# ----------------------------------------------------------------------
# 9. Detector threshold sweep on validation data (optional)
#    Use it to choose the (threshold, distance) pair for each dataset in `settings`.
# ----------------------------------------------------------------------
WEIGHTS = {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 6: 0.30}   # F_Final weights

if RUN_SWEEP:
    start, stop = split + DET_PRE + M, len(d1) - DET_POST - M
    best = {}                                              # dataset number -> (F_dataset, thr, dist)
    for n, level in enumerate(levels, 1):                  # n = 1 is D1, n = 2 is D2, ...
        sig = Functions.noisy_version(d1, level, base, rng)
        probs = Functions.detect_probs(det, sig, start, stop, mu=MU, sigma=SIGMA)
        for dist_ in (15, 25, 40):
            for thr in (0.7, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99):
                peaks, _ = find_peaks(probs, height=thr, distance=dist_)
                Index = peaks + start
                Index = Index[Index < len(sig) - CLS_POST]  # classifier window must fit
                cls = Functions.classify(sig, Index, clf, MU, SIGMA)
                f_id, f_cl, f_ds = Functions.dataset_score(Index, cls, va_Index1, va_Class1 - 1)
                print(f'D{n} noise {level:.3f} | dist {dist_} | thr {thr} | '
                      f'F_ident {f_id:.3f} F_class {f_cl:.3f} F_dataset {f_ds:.3f}')
                if n >= 2 and f_ds > best.get(n, (-1,))[0]:
                    best[n] = (f_ds, thr, dist_)

    print('\nBest per dataset:')
    for n in sorted(best):
        print(f'D{n}: F_dataset {best[n][0]:.3f} at thr {best[n][1]}, dist {best[n][2]}')
    print(f'Estimated F_Final: {sum(WEIGHTS[n] * best[n][0] for n in WEIGHTS):.3f}')

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