#!/usr/bin/env python3
"""
Trains ONE global model (FedAvg across the usable real PhysioNet patients, the
same clients/splits/hyperparameters as fedavg_vs_fedper_real.py) and exports
everything the ESP32 needs to run it and check itself against Python:

  model_export.h    the 289 trained weights + the feature scaling constants
                    (mean/std fitted on pooled TRAIN data only). These are the
                    same constants a live sensor reading must be scaled with.
  validation_set.h  every patient's held-out TEST windows as RAW (unscaled)
                    features, their labels, and the probability Python's own
                    float32 forward pass gives each window.

esp32_model_validate.ino then scales the raw windows itself, runs the model,
and compares its probabilities and per-patient balanced accuracy with
Python's. If they agree, the board computes exactly what Python computes.

    python export_model_for_esp32.py apnea_ecg_features.csv --out esp32_model_validate

Needs train_on_real_apnea.py, fedavg_vs_fedper_real.py and spec.py in the
same folder (you already have them).
"""
import argparse
import os
import struct
import urllib.request

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_fit, zscore_apply, forward, evaluate
from fedavg_vs_fedper_real import pick_clients


def cfloat(v):
    """Exact float32 round-trip as a C literal ('0' -> '0.0f', never '0f')."""
    s = f"{float(v):.9g}"
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"


def rows_of(values, per_line=8):
    out = []
    for i in range(0, len(values), per_line):
        out.append("  " + ", ".join(cfloat(v) for v in values[i:i + per_line]))
    return ",\n".join(out)


def fetch_weights(url):
    """Download the server's current global model (the aggregated result of the last finished round)."""
    with urllib.request.urlopen(url.rstrip("/") + "/model?node=exporter", timeout=15) as r:
        body = r.read()
    magic, rnd, n = struct.unpack_from("<4sIH", body, 0)
    if magic != b"FLM1":
        raise SystemExit("that server did not answer like fl_server.py")
    if n != spec.N_TOTAL:
        raise SystemExit(f"the server holds only {n} shared parameters, not the full {spec.N_TOTAL}: it is in "
                         f"FedPer mode (each node keeps a private final layer). Restart it with --mode fedavg "
                         f"and run the federation again.")
    w = np.frombuffer(body[10:], dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(w)):
        raise SystemExit("the server's weights contain NaN/inf")
    return w, rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="apnea_ecg_features.csv")
    ap.add_argument("--out", default="esp32_model_validate")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-server", default=None,
                    help="e.g. http://<pi-ip>:8080 - use the weights the federation trained on that server "
                         "(it must be in fedavg mode and still running) instead of training in Python")
    args = ap.parse_args()

    records, y, X = load_csv(args.csv)
    clients, excluded = pick_clients(records, y, X)
    print(f"{len(clients)} clients: {[str(c.record) for c in clients]}")
    if len(clients) < 2:
        raise SystemExit("need at least 2 usable clients - check the CSV")

    # scaling constants from pooled TRAIN rows only (never test rows)
    all_train = np.concatenate([c.Xtr_raw for c in clients])
    mu, sd = zscore_fit(all_train)
    for c in clients:
        c.apply_norm(mu, sd)

    if args.from_server:
        global_w, srv_round = fetch_weights(args.from_server)
        source = f"fetched from {args.from_server} after federated round {srv_round}"
        print(f"model weights {source}")
    else:
        # FedAvg simulated in Python: whole model shared, sample-count-weighted, same as fl_server.py
        rng = np.random.default_rng(args.seed)
        n_shared = spec.N_TOTAL
        global_w = spec.init_params(rng)[:n_shared].copy()
        for r in range(args.rounds):
            ups = [c.local_round(global_w, n_shared, rng) for c in clients]
            ns = np.array([u[1] for u in ups], dtype=np.float64)
            W = np.stack([u[0] for u in ups])
            global_w = (W * (ns / ns.sum())[:, None]).sum(axis=0).astype(np.float32)
        source = f"{args.rounds} FedAvg rounds simulated in Python, seed {args.seed}"
        print(f"trained global model: {source}")

    names, pid, lab, xraw, ppy, balacc = [], [], [], [], [], []
    for k, c in enumerate(clients):
        names.append(str(c.record))
        Xn = zscore_apply(c.Xte_raw, mu, sd).astype(np.float32)
        p = forward(global_w, Xn).reshape(-1).astype(np.float32)
        ev = evaluate(global_w, Xn, c.yte)
        balacc.append(ev["balanced_accuracy"])
        for i in range(len(c.yte)):
            pid.append(k)
            lab.append(int(c.yte[i]))
            xraw.append(c.Xte_raw[i])
            ppy.append(p[i])
        print(f"  {names[-1]}: {len(c.yte):3d} test windows, "
              f"Python balanced_acc={ev['balanced_accuracy']:.4f}")
    n = len(lab)
    mean_bal = float(np.mean(balacc))
    print(f"Python mean balanced accuracy over {len(clients)} patients: {mean_bal:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model_export.h"), "w") as f:
        f.write("// generated by export_model_for_esp32.py - do not edit\n")
        f.write(f"// weights: {source}; clients {names}\n")
        f.write("#define MODEL_N_PARAMS 289\n")
        f.write("static const float MODEL_W[289] = {\n" + rows_of(global_w) + "\n};\n")
        f.write("static const float FEAT_MU[8] = {\n" + rows_of(mu) + "\n};\n")
        f.write("static const float FEAT_SD[8] = {\n" + rows_of(sd) + "\n};\n")

    with open(os.path.join(args.out, "validation_set.h"), "w") as f:
        f.write("// generated by export_model_for_esp32.py - do not edit\n")
        f.write("#include <stdint.h>\n")
        f.write(f"#define VAL_N {n}\n#define VAL_N_PATIENTS {len(clients)}\n")
        f.write("static const char* const VAL_PATIENT_NAMES[VAL_N_PATIENTS] = {"
                + ", ".join(f'"{s}"' for s in names) + "};\n")
        f.write("static const float VAL_PY_BALACC[VAL_N_PATIENTS] = {"
                + ", ".join(cfloat(v) for v in balacc) + "};\n")
        f.write(f"static const float VAL_PY_MEAN_BALACC = {cfloat(mean_bal)};\n")
        f.write("static const uint8_t VAL_PID[VAL_N] = {"
                + ",".join(str(v) for v in pid) + "};\n")
        f.write("static const uint8_t VAL_Y[VAL_N] = {"
                + ",".join(str(v) for v in lab) + "};\n")
        f.write("static const float VAL_P_PY[VAL_N] = {\n" + rows_of(ppy) + "\n};\n")
        f.write("static const float VAL_X[VAL_N][8] = {\n")
        f.write(",\n".join("  {" + ", ".join(cfloat(v) for v in row) + "}" for row in xraw))
        f.write("\n};\n")

    print(f"\nwrote {args.out}/model_export.h and {args.out}/validation_set.h "
          f"({n} test windows)")


if __name__ == "__main__":
    main()
