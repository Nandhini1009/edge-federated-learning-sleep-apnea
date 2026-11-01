#!/usr/bin/env python3
"""
Train the SAME 8-16-8-1 MLP (spec.py) on REAL PhysioNet Apnea-ECG features,
with a proper patient-level (by-record) train/test split so no minute from a
test patient ever leaked into training - this is the number that actually
belongs in your report, not the placeholder-label scores from the ESP32 demo.

    python train_on_real_apnea.py

Requires apnea_ecg_features.csv from apnea_ecg_extract.py.
"""
import csv
import sys

import numpy as np

import spec  # reuses your frozen model layout: 8 -> 16 -> 8 -> 1


def load_csv(path):
    records, labels, feats = [], [], []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            records.append(row[0])
            labels.append(int(row[1]))
            feats.append([float(x) for x in row[2:]])
    return np.array(records), np.array(labels, dtype=np.float32), np.array(feats, dtype=np.float32)


def split_by_record(records, test_records):
    is_test = np.isin(records, test_records)
    return ~is_test, is_test


def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1e-6
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def forward(params, X):
    layers = spec.unpack(params)
    a = X
    for i, (W, b) in enumerate(layers):
        z = a @ W.T + b
        a = np.maximum(z, 0.0) if i < len(layers) - 1 else 1.0 / (1.0 + np.exp(-z))
    return a


def train_epoch(params, X, y, lr, batch, pos_weight, rng):
    layers = spec.unpack(params)
    idx = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        bi = idx[s:s + batch]
        yb = y[bi].reshape(-1, 1)
        acts = [X[bi]]
        a = X[bi]
        for i, (W, b) in enumerate(layers):
            z = a @ W.T + b
            a = np.maximum(z, 0.0) if i < len(layers) - 1 else 1.0 / (1.0 + np.exp(-z))
            acts.append(a)
        p = acts[-1]
        dz = (p * (1.0 - yb) - pos_weight * yb * (1.0 - p)) / len(bi)
        for l in range(len(layers) - 1, -1, -1):
            W, b = layers[l]
            gW = dz.T @ acts[l]
            gb = dz.sum(axis=0)
            if l > 0:
                dz_next = (dz @ W) * (acts[l] > 0.0)
            W -= lr * gW
            b -= lr * gb
            if l > 0:
                dz = dz_next


def evaluate(params, X, y, thresh=0.5):
    p = forward(params, X).reshape(-1)
    pred = (p >= thresh).astype(np.float32)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tpr = tp / max(tp + fn, 1)   # sensitivity / recall
    tnr = tn / max(tn + fp, 1)   # specificity
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-9)
    bal_acc = 0.5 * (tpr + tnr)
    return dict(balanced_accuracy=bal_acc, sensitivity=tpr, specificity=tnr,
                precision=precision, f1=f1, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "apnea_ecg_features.csv"
    records, y, X = load_csv(path)
    uniq = sorted(set(records))
    print(f"loaded {len(y)} rows from {len(uniq)} records: {uniq}")
    print(f"overall class balance: {int(y.sum())} apnea / {int(len(y) - y.sum())} normal")

    # Patient-level split: hold out whichever records you pass, train on the rest.
    # Default: hold out the last record alphabetically as a quick single-patient
    # test; for a real result use several held-out records (see the printed
    # per-record breakdown to choose a reasonable split for what you downloaded).
    test_records = sys.argv[2].split(",") if len(sys.argv) > 2 else [uniq[-1]]
    train_mask, test_mask = split_by_record(records, test_records)
    print(f"train records: {sorted(set(records[train_mask]))}  ({train_mask.sum()} rows)")
    print(f"test  records: {sorted(set(records[test_mask]))}  ({test_mask.sum()} rows)")
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("ERROR: empty train or test split - check the record names you passed.")
        return

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    mu, sd = zscore_fit(Xtr)          # fit normalization on TRAIN ONLY
    Xtr = zscore_apply(Xtr, mu, sd)
    Xte = zscore_apply(Xte, mu, sd)

    rng = np.random.default_rng(0)
    params = spec.init_params(rng)
    n_pos = ytr.sum()
    pos_weight = float((len(ytr) - n_pos) / max(n_pos, 1))
    print(f"train pos_weight={pos_weight:.3f} (class imbalance correction)")

    EPOCHS = 60
    for e in range(1, EPOCHS + 1):
        train_epoch(params, Xtr, ytr, lr=0.05, batch=32, pos_weight=pos_weight, rng=rng)
        if e % 10 == 0 or e == EPOCHS:
            tr = evaluate(params, Xtr, ytr)
            te = evaluate(params, Xte, yte)
            print(f"epoch {e:3d}  train balanced_acc={tr['balanced_accuracy']:.3f}  "
                  f"test balanced_acc={te['balanced_accuracy']:.3f}  test F1={te['f1']:.3f}")

    print("\n=== FINAL TEST RESULT (held-out patient(s), never seen in training) ===")
    final = evaluate(params, Xte, yte)
    for k, v in final.items():
        print(f"  {k}: {v}")
    print("\nSave these numbers with the exact train/test record split - that split is "
          "the single most important methodological detail for anyone reproducing this.")


if __name__ == "__main__":
    main()
