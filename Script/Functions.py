#Imports
import numpy as np
import torch
import scipy.io as spio
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks, correlate
from scipy.ndimage import median_filter

from Config import DET_PRE, DET_POST, CLS_PRE, CLS_POST, DEVICE
device = DEVICE
def load_data(data_dir):
    """Load D1 (with labels) and D2-D6 (raw only). Returns (ds, Index1, Class1)."""
    data_dir = Path(data_dir)
    mat = spio.loadmat(data_dir / 'D1.mat', squeeze_me=True)
    ds = [mat['d']]
    Index1, Class1 = mat['Index'], mat['Class']
    for i in range(2, 7):
        ds.append(spio.loadmat(data_dir / f'D{i}.mat', squeeze_me=True)['d'])
    return ds, Index1, Class1

#Noise function
def noise_std(sig):
    return 1.4826 * np.median(np.abs(sig - np.median(sig)))   # robust, ignores sparse spikes

def noisy_version(sig, target, base, rng):
    extra = np.sqrt(max(target**2 - base**2, 0))
    return sig + rng.normal(0, extra, sig.shape)

def add_noise(xb, levels_extra, weights=None):
    """Add fresh Gaussian noise at a randomly chosen level.

    weights: optional tensor with one sampling weight per level (need not sum to 1).
             None = all levels equally likely.
    """
    if weights is None:
        idx = torch.randint(len(levels_extra), (len(xb),), device=xb.device)
    else:
        idx = torch.multinomial(weights, len(xb), replacement=True)
    sigma = levels_extra[idx].view(-1, 1, 1)
    return xb + sigma * torch.randn_like(xb)

def make_loader(X, y, shuffle, mu, sigma):
    X = torch.tensor(((X - mu) / sigma)[:, None, :], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=shuffle)

#Windowing function
def windows(sig, positions, pre, post, jitter=0, rng=None):
    """Cut a window around each position, optionally shifted by random jitter."""
    out = []
    for t in positions:
        if jitter:
            t = t + rng.integers(-jitter, jitter + 1)
        out.append(sig[t - pre : t + post])
    return np.array(out)

#Hard negative selection function
def hard_negatives(idx_sub, lo, hi, all_idx, rng):
    offs = rng.integers(15, 61, len(idx_sub)) * rng.choice([-1, 1], len(idx_sub))
    pos = idx_sub + offs
    pos = pos[(pos >= lo) & (pos < hi)]
    nearest = np.abs(pos[:, None] - all_idx[None, :]).min(axis=1)
    return pos[nearest >= 15]          # drop any that land on a neighbouring spike

#Accuracy function
def accuracy(model, X, y, mu, sigma):
    X = torch.tensor(((X - mu) / sigma)[:, None, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        return (model(X).argmax(1).cpu().numpy() == y).mean()

#training function
def train_model(model, train_dl, val_dl, loss_fn, epochs=40, lr=1e-3, patience=8, augment=None):
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=3)

    best_loss, best_state, epochs_no_improve = np.inf, None, 0

    for epoch in range(epochs):
        # ---- training pass ----
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            if augment is not None:
                xb = augment(xb)          # fresh noise every batch
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        # ---- validation pass ----
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += loss_fn(out, yb).item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
        n = len(val_dl.dataset)
        val_loss /= n
        scheduler.step(val_loss)
        print(f'epoch {epoch + 1:2d} | val loss {val_loss:.4f} | val acc {correct / n:.3f}')

        # ---- early stopping: keep the best weights ----
        if val_loss < best_loss - 1e-4:
            best_loss, epochs_no_improve = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model

# functions for detector evaluation
def detect_probs(model, sig, start, stop,mu, sigma, batch=8192):
    """P(spike starts at t) for every t in [start, stop)."""
    L = DET_PRE + DET_POST
    view = sliding_window_view(sig, L)           # view[k] = sig[k : k+L], so t = k + DET_PRE
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(start - DET_PRE, stop - DET_PRE, batch):
            xb = view[k : min(k + batch, stop - DET_PRE)]
            xb = torch.tensor(((xb - mu) / sigma)[:, None, :], dtype=torch.float32).to(device)
            out.append(torch.softmax(model(xb), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def match_f1(pred, true, tol=50):
    """One-to-one matching within +/-tol samples, like the automarker."""
    pred, used, tp = np.sort(pred), np.zeros(len(pred), bool), 0
    for t in np.sort(true):
        j = np.searchsorted(pred, t)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(pred) and not used[k] and abs(pred[k] - t) <= tol:
                if best is None or abs(pred[k] - t) < abs(pred[best] - t):
                    best = k
        if best is not None:
            used[best], tp = True, tp + 1
    p, r = tp / max(len(pred), 1), tp / max(len(true), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# functions for classifier and detection model application to dataset
def predict_dataset(sig, thr, dist_, det, clf, mu, sigma, mf_template=None, mf_z=None, mf_gap=40):
    """Detect spikes in a recording, then classify each one. Returns (Index, Class)."""
    start, stop = CLS_PRE, len(sig) - CLS_POST        # keeps classifier windows inside the signal
    probs = detect_probs(det, sig, start, stop, mu, sigma)
    peaks, _ = find_peaks(probs, height=thr, distance=dist_)
    Index = peaks + start

    if mf_template is not None:
        score = matched_filter_score(sig, mf_template)
        cand, _ = find_peaks(score, height=mf_z, distance=dist_)
        cand = cand[(cand >= CLS_PRE) & (cand < len(sig) - CLS_POST)]
        Index = add_far_candidates(Index, cand, mf_gap)

    X = windows(sig, Index, CLS_PRE, CLS_POST)        # no jitter at inference
    X = torch.tensor(((X - mu) / sigma)[:, None, :], dtype=torch.float32)
    clf.eval()
    with torch.no_grad():
        #Class = np.concatenate([clf(X[i:i + 4096].to(device)).argmax(1).cpu().numpy()
                               # for i in range(0, len(X), 4096)]) + 1   # back to labels 1-5
        Class = classify(sig, Index, clf, mu, sigma) + 1      # back to labels 1-5
    return Index, Class

def classify(sig, Index, clf, mu, sigma, batch=4096):
    """Class (0-4) of the spike starting at each position in Index."""
    if len(Index) == 0:
        return np.array([], dtype=int)
    X = windows(sig, Index, CLS_PRE, CLS_POST)
    X = torch.tensor(((X - mu) / sigma)[:, None, :], dtype=torch.float32)
    clf.eval()
    with torch.no_grad():
        return np.concatenate([clf(X[i:i + batch].to(DEVICE)).argmax(1).cpu().numpy()
                               for i in range(0, len(X), batch)])


def match_pairs(pred, true, tol=50):
    """One-to-one matching within +/-tol samples. Returns a list of (pred_i, true_j)."""
    order = np.argsort(pred)
    pred_s = pred[order]
    used = np.zeros(len(pred), bool)
    pairs = []
    for jt in np.argsort(true):
        t = true[jt]
        j = np.searchsorted(pred_s, t)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(pred_s) and not used[k] and abs(pred_s[k] - t) <= tol:
                if best is None or abs(pred_s[k] - t) < abs(pred_s[best] - t):
                    best = k
        if best is not None:
            used[best] = True
            pairs.append((order[best], jt))
    return pairs


def dataset_score(pred_idx, pred_cls, true_idx, true_cls, tol=50):
    """Returns (F_ident, F_class, F_dataset = 0.3*F_ident + 0.7*F_class)."""
    pairs = match_pairs(pred_idx, true_idx, tol)
    found = len(pairs)
    correct = sum(pred_cls[i] == true_cls[j] for i, j in pairs)

    def f1(hits):
        p = hits / max(len(pred_idx), 1)
        r = hits / max(len(true_idx), 1)
        return 2 * p * r / max(p + r, 1e-9)

    f_ident, f_class = f1(found), f1(correct)
    return f_ident, f_class, 0.3 * f_ident + 0.7 * f_class

# Drift removal function
def remove_baseline(sig, size=2501):
    """Subtract the slow baseline (about 0.1 s median) from a recording."""
    return sig - median_filter(sig, size=size)

#template classifier
def class_templates(sig, positions, labels, pre, post):
    """Mean window for each class (labels 1-5). Returns an array of shape (5, pre + post)."""
    W = windows(sig, positions, pre, post)
    return np.array([W[labels == k].mean(axis=0) for k in range(1, 6)])

def template_classify(X, templates):
    """Class 0-4 of each window = nearest template in squared distance."""
    d = ((X[:, None, :] - templates[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)

#matched filter         
def class_template(sig, positions, labels, k, length=60):
    """Mean waveform of class k, `length` samples from each spike start."""
    W = windows(sig, positions[labels == k], 0, length)
    return W.mean(axis=0)


def matched_filter_score(sig, template):
    """Slide the template along the recording. score[t] is in noise-std units,
    so a value of 4 means 'a spike starting at t fits 4 noise-sigmas better than nothing'."""
    corr = correlate(sig, template, mode='valid', method='fft')
    return corr / (noise_std(sig) * np.linalg.norm(template))


def add_far_candidates(index, cand, min_gap=40):
    """Keep every CNN detection and add matched-filter candidates that are more than
    min_gap samples from all of them (so one spike is not counted twice)."""
    index = np.sort(index)
    if len(cand) == 0 or len(index) == 0:
        return np.sort(np.concatenate([index, cand])).astype(int)
    j = np.searchsorted(index, cand)
    left = np.abs(cand - index[np.clip(j - 1, 0, len(index) - 1)])
    right = np.abs(cand - index[np.clip(j, 0, len(index) - 1)])
    keep = np.minimum(left, right) > min_gap
    return np.sort(np.concatenate([index, cand[keep]]))

def template_classify_shift(sig, Index, templates, shifts=range(-8, 9)):
    """Nearest-template class (0-4). Each detection is tried at several shifts and the
    best-matching (class, shift) pair decides the label."""
    best = np.full((len(Index), len(templates)), np.inf)
    for s in shifts:
        pos = np.clip(Index + s, CLS_PRE, len(sig) - CLS_POST)
        X = windows(sig, pos, CLS_PRE, CLS_POST)
        d = ((X[:, None, :] - templates[None, :, :]) ** 2).sum(axis=2)
        best = np.minimum(best, d)
    return best.argmin(axis=1)