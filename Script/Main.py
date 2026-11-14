#Main script for EE32009 Coursework 2024

#  Import libraries
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from pathlib import Path
import Functions
import Models
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from Config import DATA_DIR
import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = DEVICE

#load all data
ds, Index1, Class1 = Functions.load_data(DATA_DIR)
d1 = ds[0]


#Noise estimation function

base = Functions.noise_std(d1)   # use D1 as baseline noise level
for n, sig in enumerate(ds, 1):
    print(f'D{n} noise std: {Functions.noise_std(sig):.4f}')

targets = [Functions.noise_std(sig) for sig in ds[1:]]   # use measured D2..D6 as targets

rng = np.random.default_rng(0)


#Building the training data & validation data
DET_PRE, DET_POST = 15, 50
CLS_PRE, CLS_POST = 15, 90
M = 30                                     # margin so jitter never leaves the signal
split = int(0.8 * len(d1))                # 1,152,000


# keep spikes whose (jittered) window fits inside the recording; filter labels together
ok = (Index1 > CLS_PRE + M) & (Index1 < len(d1) - CLS_POST - M)
Index1, Class1 = Index1[ok], Class1[ok]

# split by time, leaving a gap so train and val windows don't overlap
tr = Index1 < split - CLS_POST - M
va = Index1 >= split + CLS_PRE + M
tr_Index1, tr_Class1 = Index1[tr], Class1[tr]
va_Index1, va_Class1 = Index1[va], Class1[va]

# background positions, split the same way
cand = rng.integers(CLS_PRE + M, len(d1) - CLS_POST - M, size=30000)
dist = np.abs(cand[:, None] - Index1[None, :]).min(axis=1)
cand = cand[dist > 100]
neg_tr = cand[cand < split - CLS_POST - M][:3 * len(tr_Index1)]
neg_va = cand[cand >= split + CLS_PRE + M][:3 * len(va_Index1)]

#prepare detection model training data (spikes vs background)


hn_tr = Functions.hard_negatives(tr_Index1, DET_PRE + M, split - DET_POST - M, Index1, rng)
hn_va = Functions.hard_negatives(va_Index1, split + DET_PRE + M, len(d1) - DET_POST - M, Index1, rng)

det_val = []

#stacking trainning data to make matrix
Xc, yc = [], []          # classifier: spikes only
Xd, yd = [], []          # detector: spikes (1) vs background (0)
val_sets = []            # set of validation data for each noise level

levels = [base] + targets
# training data: clean D1 windows (noise is added later, during training)
Xc = Functions.windows(d1, tr_Index1, CLS_PRE, CLS_POST, jitter=10, rng=rng)
yc = tr_Class1

Xd = np.concatenate([Functions.windows(d1, tr_Index1, DET_PRE, DET_POST, jitter=10, rng=rng),
                     Functions.windows(d1, neg_tr, DET_PRE, DET_POST),
                     Functions.windows(d1, hn_tr, DET_PRE, DET_POST)])
yd = np.concatenate([np.ones(len(tr_Index1)),
                     np.zeros(len(neg_tr)),
                     np.zeros(len(hn_tr))])

for level in levels:
    sig = Functions.noisy_version(d1, level, base, rng)

    # classifier training data (spikes only)
    #Xc.append(Functions.windows(sig, tr_Index1, CLS_PRE, CLS_POST, jitter=10));
    #yc.append(tr_Class1)

    # detector training data (spikes, background, hard negatives)
    #Xd.append(Functions.windows(sig, tr_Index1, DET_PRE, DET_POST, jitter=10));  yd.append(np.ones(len(tr_Index1)))
    #Xd.append(Functions.windows(sig, neg_tr, DET_PRE, DET_POST));                yd.append(np.zeros(len(neg_tr)))
    #Xd.append(Functions.windows(sig, hn_tr, DET_PRE, DET_POST));                 yd.append(np.zeros(len(hn_tr)))

    # detector validation data
    Xv = np.concatenate([Functions.windows(sig, va_Index1, DET_PRE, DET_POST),
                         Functions.windows(sig, neg_va, DET_PRE, DET_POST),
                         Functions.windows(sig, hn_va, DET_PRE, DET_POST)])
    yv = np.concatenate([np.ones(len(va_Index1)), np.zeros(len(neg_va)), np.zeros(len(hn_va))])
    det_val.append((Xv, yv))

    # classifier validation data, kept per noise level
    val_sets.append({'level': level,
                     'X': Functions.windows(sig, va_Index1, CLS_PRE, CLS_POST),
                     'y': va_Class1 - 1})

Xc_va = np.concatenate([v['X'] for v in val_sets])
yc_va = np.concatenate([v['y'] for v in val_sets])

#machine learning model
torch.manual_seed(0)
MU, SIGMA = float(np.median(d1)), float(d1.std())

#extra noise 
levels_extra = torch.tensor(
    [np.sqrt(max(l**2 - base**2, 0)) for l in levels],
    dtype=torch.float32, device=device) / SIGMA



#training loop

augment = lambda xb: Functions.add_noise(xb, levels_extra)
train_dl = Functions.make_loader(Xc, yc.astype(int) - 1, shuffle=True, mu=MU, sigma=SIGMA)
val_dl = Functions.make_loader(Xc_va, yc_va, shuffle=False, mu=MU, sigma=SIGMA)

clf = Models.SpikeCNN(n_out=5)
clf = Functions.train_model(clf, train_dl, val_dl, nn.CrossEntropyLoss(), augment=augment)

#validating the model on D1-D6
model = clf.eval()

#check overfitting by calculating accuracy on training data

for v in val_sets:
    X = torch.tensor(((v['X'] - MU) / SIGMA)[:, None, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()
    acc = (pred == v['y']).mean()
    sig = Functions.noisy_version(d1, v['level'], base, rng)
    Xj = Functions.windows(sig, va_Index1, CLS_PRE, CLS_POST, jitter=10, rng=rng)
    print(f"jittered window: noise std {v['level']:.3f} | acc {Functions.accuracy(clf, Xj, v['y'], MU, SIGMA):.3f}")
    print(f"aligned window: noise std {v['level']:.3f} | acc {acc:.3f}")
    print(confusion_matrix(v['y'], pred))


#detection model

det_dl = Functions.make_loader(Xd, yd, shuffle=True, mu=MU, sigma=SIGMA)
det_val_dl = Functions.make_loader(np.concatenate([x for x, _ in det_val]),
                         np.concatenate([y for _, y in det_val]), shuffle=False, mu=MU, sigma=SIGMA)

det = Models.SpikeCNN(n_out=2)
det = Functions.train_model(det, det_dl, det_val_dl, nn.CrossEntropyLoss(), augment=augment)

start, stop = split + DET_PRE + M, len(d1) - DET_POST - M
for level in levels:
    sig = Functions.noisy_version(d1, level, base, rng)
    probs = Functions.detect_probs(det, sig, start, stop, mu=MU, sigma=SIGMA)
    for dist_ in (15, 25, 40):
        for thr in (0.85, 0.9, 0.93, 0.95, 0.97, 0.98):
            peaks, _ = find_peaks(probs, height=thr, distance=dist_)
            p, r, f1 = Functions.match_f1(peaks + start, va_Index1)
            print(f'noise {level:.3f} | dist {dist_} | thr {thr} | Precision {p:.3f} Recall {r:.3f} F1 {f1:.3f}')




# starting settings, tune per dataset from your validation sweep and automarker feedback
settings = {2: (0.7, 15), 3: (0.85, 15), 4: (0.93, 25), 5: (0.9, 25), 6: (0.85, 15)}

out_dir = Path('submission'); out_dir.mkdir(exist_ok=True)
print(f'D1 reference: {len(Index1)} spikes')
for n in range(2, 7):
    thr, dist_ = settings[n]
    Index, Class = Functions.predict_dataset(ds[n - 1], thr, dist_, det, clf, MU, SIGMA)
    print(f'D{n}: {len(Index)} spikes, class counts {np.bincount(Class, minlength=6)[1:]}')
    spio.savemat(out_dir / f'D{n}.mat', {'Index': Index.astype(np.int64), 'Class': Class.astype(np.int64)})

#check the saved files
m = spio.loadmat('submission/D2.mat', squeeze_me=True)
print(m.keys())          # should show Index and Class (plus a few __header__ entries)