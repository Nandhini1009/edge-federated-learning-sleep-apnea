#!/usr/bin/env python3
"""
Exports ONE real patient's HRV features + labels from apnea_ecg_features.csv
into a simple ASCII format the ESP32 can parse over Serial, using the exact
same preprocessing as fedavg_vs_fedper_real.py (time-blocked 70/30 split,
pooled-training z-score normalization, per-patient pos_weight) - so the
ESP32's resulting score is directly comparable to that script's per-client
result for the same patient.

    python export_patient_for_esp32.py apnea_ecg_features.csv b05

Writes <record>_for_esp32.txt, lines:
    N_TRAIN n_train
    N_TEST n_test
    POS_WEIGHT <float>
    TRAIN f0 f1 f2 f3 f4 f5 f6 f7 label
    ... (n_train lines)
    TEST f0 f1 f2 f3 f4 f5 f6 f7 label
    ... (n_test lines)
    DONE

Also prints the Python-reference balanced accuracy a plain (non-federated,
single-client) run of this exact data gets, as a sanity baseline for
comparison - the ESP32 result should land close to this when it talks to the
same server the other clients aren't also updating, or close to the
federated per-client score from fedavg_vs_fedper_real.py's output when
run as one of several real nodes.
"""
import sys

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_fit, zscore_apply, train_epoch, evaluate
from fedavg_vs_fedper_real import pick_clients, pooled_scaling


def main():
    if len(sys.argv) < 3:
        print("usage: python export_patient_for_esp32.py apnea_ecg_features.csv <record>")
        sys.exit(1)
    path, record = sys.argv[1], sys.argv[2]
    local_scaling = "--local-scaling" in sys.argv

    records, y, X = load_csv(path)
    mask = records == record
    if mask.sum() == 0:
        print(f"record '{record}' not found. Available: {sorted(set(records))}")
        sys.exit(1)
    Xr, yr = X[mask], y[mask]
    cut = int(0.7 * len(Xr))
    Xtr_raw, ytr = Xr[:cut], yr[:cut]
    Xte_raw, yte = Xr[cut:], yr[cut:]

    if local_scaling:
        mu, sd = zscore_fit(Xtr_raw)
        print("scaling: LOCAL - this patient's own statistics (old behaviour)")
    else:
        all_clients, _ = pick_clients(records, y, X)
        if record not in [str(c.record) for c in all_clients]:
            print(f"warning: {record} is not one of the usable patients {[str(c.record) for c in all_clients]}")
        mu, sd = pooled_scaling(all_clients)
        print(f"scaling: SHARED - one mean/std pooled over {len(all_clients)} patients; the simulated "
              f"patients and the model export use these same constants")
    Xtr = zscore_apply(Xtr_raw, mu, sd)
    Xte = zscore_apply(Xte_raw, mu, sd)

    n_pos = ytr.sum()
    pos_weight = float((len(ytr) - n_pos) / max(n_pos, 1))

    out_path = f"{record}_for_esp32.txt"
    with open(out_path, "w") as f:
        f.write(f"N_TRAIN {len(Xtr)}\n")
        f.write(f"N_TEST {len(Xte)}\n")
        f.write(f"POS_WEIGHT {pos_weight:.6f}\n")
        for x, lab in zip(Xtr, ytr):
            f.write("TRAIN " + " ".join(f"{v:.6f}" for v in x) + f" {int(lab)}\n")
        for x, lab in zip(Xte, yte):
            f.write("TEST " + " ".join(f"{v:.6f}" for v in x) + f" {int(lab)}\n")
        f.write("DONE\n")

    print(f"wrote {out_path}: {len(Xtr)} train, {len(Xte)} test, pos_weight={pos_weight:.3f}")

    # Python-reference sanity baseline: a standalone (non-federated) run on
    # exactly this exported data, so you have something to compare the ESP32
    # number against even before it talks to the real aggregator
    rng = np.random.default_rng(0)
    params = spec.init_params(rng)
    idx = np.arange(len(Xtr))
    for _ in range(3):   # same LOCAL_EPOCHS as esp32_fl_node.ino / fedavg_vs_fedper_real.py
        rng.shuffle(idx)
        train_epoch(params, Xtr[idx], ytr[idx], lr=0.05, batch=32, pos_weight=pos_weight, rng=rng)
    score = evaluate(params, Xte, yte)
    print(f"Python standalone reference (3 local epochs, same data/hyperparameters): "
          f"balanced_accuracy={score['balanced_accuracy']:.3f}")
    print("(this uses a fresh random init, same as the ESP32 will when it starts cold - "
          "expect the ESP32's number to be in the same ballpark, not bit-identical, since "
          "federation with the other real nodes on the server will also influence its result)")


if __name__ == "__main__":
    main()
