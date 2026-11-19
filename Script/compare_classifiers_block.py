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
        m = Models.SpikeCNN(n_out=5, lowpass_hz=LOWPASS_HZ)
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
