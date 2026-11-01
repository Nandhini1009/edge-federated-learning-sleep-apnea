#!/usr/bin/env python3
"""
Leave-one-record-out cross-validation on real PhysioNet Apnea-ECG features.
Answers the question the single b01 test result couldn't: how much does this
model's accuracy vary depending on WHICH patient gets held out?

    python train_on_real_apnea_cv.py apnea_ecg_features.csv

Reuses every function from train_on_real_apnea.py unchanged - this script
only adds the fold loop and the summary statistics across folds.
"""
import sys

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_fit, zscore_apply, train_epoch, evaluate


def run_fold(records, y, X, test_record, epochs=60, seed=0):
    train_mask = records != test_record
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    mu, sd = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, sd)
    Xte = zscore_apply(Xte, mu, sd)

    rng = np.random.default_rng(seed)
    params = spec.init_params(rng)
    n_pos = ytr.sum()
    pos_weight = float((len(ytr) - n_pos) / max(n_pos, 1))
    for _ in range(epochs):
        train_epoch(params, Xtr, ytr, lr=0.05, batch=32, pos_weight=pos_weight, rng=rng)

    result = evaluate(params, Xte, yte)
    result["n_train"] = int(train_mask.sum())
    result["n_test"] = int(test_mask.sum())
    result["n_test_pos"] = int(yte.sum())
    result["train_pos_weight"] = pos_weight
    return result


MIN_TEST_SIZE = 20   # folds with fewer test rows than this are shown but excluded
                      # from the summary - their estimate is too noisy to trust
                      # (e.g. a record where extraction recovered only 2-14
                      # usable minutes due to a long unresolved signal gap)

MIN_TEST_POS = 10     # folds with fewer than this many POSITIVE (apnea) test
                       # rows are ALSO excluded - not because the sample is
                       # small, but because balanced_accuracy is structurally
                       # biased when one class is near-absent: with 0 true
                       # positives possible, sensitivity is forced to exactly
                       # 0 regardless of the model, which mechanically halves
                       # balanced_accuracy to 0.5*specificity. A control
                       # record with 0 apnea minutes (e.g. c03 in this
                       # project's data) will ALWAYS score badly on this
                       # metric even with a perfect model - that is a property
                       # of the metric on that fold, not a real result.


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "apnea_ecg_features.csv"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    records, y, X = load_csv(path)
    uniq = sorted(set(records))
    print(f"loaded {len(y)} rows from {len(uniq)} records: {uniq}")
    print(f"overall class balance: {int(y.sum())} apnea / {int(len(y) - y.sum())} normal")
    print(f"running leave-one-record-out CV, {epochs} epochs per fold "
          f"(folds with <{MIN_TEST_SIZE} test rows are excluded from the summary)\n")

    results = []
    excluded = []
    for rec in uniq:
        r = run_fold(records, y, X, rec, epochs=epochs)
        if r is None:
            print(f"  {rec}: skipped (empty split)")
            continue
        flag = ""
        if r["n_test"] < MIN_TEST_SIZE:
            excluded.append((rec, r))
            flag = f"  [EXCLUDED from summary: n_test < {MIN_TEST_SIZE}]"
        elif r["n_test_pos"] < MIN_TEST_POS:
            excluded.append((rec, r))
            flag = (f"  [EXCLUDED from summary: only {r['n_test_pos']} apnea minutes in "
                     f"test - balanced_accuracy is structurally biased near 0.5*specificity "
                     f"when a class is this sparse/absent]")
        else:
            results.append((rec, r))
        print(f"  held out {rec:>4s} (n_test={r['n_test']:4d}, {r['n_test_pos']:3d} apnea): "
              f"balanced_acc={r['balanced_accuracy']:.3f}  sens={r['sensitivity']:.3f}  "
              f"spec={r['specificity']:.3f}  F1={r['f1']:.3f}{flag}")

    if not results:
        print("no folds with enough test data ran - check your record names / CSV.")
        return

    bal_accs = np.array([r["balanced_accuracy"] for _, r in results])
    sens = np.array([r["sensitivity"] for _, r in results])
    spec_ = np.array([r["specificity"] for _, r in results])
    f1s = np.array([r["f1"] for _, r in results])

    if excluded:
        print(f"\n({len(excluded)} fold(s) excluded from the summary - too few test "
              f"rows and/or too few apnea examples to trust: {[r for r, _ in excluded]})")
    print(f"\n=== SUMMARY across {len(results)} held-out patients ===")
    print(f"  balanced_accuracy: {bal_accs.mean():.3f} +/- {bal_accs.std():.3f}  "
          f"(min={bal_accs.min():.3f}, max={bal_accs.max():.3f})")
    print(f"  sensitivity:       {sens.mean():.3f} +/- {sens.std():.3f}")
    print(f"  specificity:       {spec_.mean():.3f} +/- {spec_.std():.3f}")
    print(f"  F1:                {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print("\nThis mean +/- std across held-out patients is the number to put in your")
    print("report, not any single fold - the spread itself is a real finding (it shows")
    print("how much this model's performance depends on which patient it sees).")


if __name__ == "__main__":
    main()
