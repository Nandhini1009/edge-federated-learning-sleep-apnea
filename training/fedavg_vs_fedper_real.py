#!/usr/bin/env python3
"""
FedAvg vs FedPer, using REAL PhysioNet Apnea-ECG patients as federated
clients - this is the experiment that actually belongs in the paper, not a
synthetic stand-in.

    python fedavg_vs_fedper_real.py apnea_ecg_features.csv

Each usable patient (see MIN_TEST_SIZE/MIN_POS filtering, same criteria as
train_on_real_apnea_cv.py) becomes one federated client:
  - time-blocked 70/30 train/test split within that patient's own minutes
    (never shuffled - avoids leaking adjacent-minute correlation across the
    split, per this project's stated methodology)
  - per-client pos_weight (patients have very different apnea prevalence)
  - feature normalization uses FIXED constants computed once from the pooled
    TRAIN split of every client (never test data) - matches PROTOCOL.md's
    deployment guidance: "every node applies the same constants"

Per round: each client fetches the current global weights, trains locally
(same hyperparameters as sim_nodes.py / esp32_fl_node.ino: lr=0.05, batch=32,
3 local epochs), evaluates on its own held-out test split, and reports.
  FedAvg : the whole 289-param model is shared and averaged every round.
  FedPer : only the 280-param base is shared/averaged; each client keeps its
           own 9-param head private throughout.
Aggregation is sample-count-weighted, matching fl_server.py.

Reports per-round mean test balanced accuracy across clients (equal
per-patient weighting, matching the leave-one-record-out CV framing already
used in this project) for both modes, plus the exact bytes/round each mode
moves - the real-data version of the protocol numbers in PROTOCOL.md.
"""
import sys

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_fit, zscore_apply, forward, train_epoch, evaluate

MIN_TEST_SIZE = 20
MIN_TEST_POS = 10
ROUNDS = 20
LOCAL_EPOCHS = 3
LR = 0.05
BATCH = 32


class Client:
    def __init__(self, record, X, y):
        self.record = record
        cut = int(0.7 * len(X))   # time-blocked: first 70% train, last 30% test
        self.Xtr_raw, self.ytr = X[:cut], y[:cut]
        self.Xte_raw, self.yte = X[cut:], y[cut:]
        n_pos = self.ytr.sum()
        self.pos_weight = float((len(self.ytr) - n_pos) / max(n_pos, 1))

    def apply_norm(self, mu, sd):
        self.Xtr = zscore_apply(self.Xtr_raw, mu, sd)
        self.Xte = zscore_apply(self.Xte_raw, mu, sd)

    def local_round(self, global_shared, n_shared, rng):
        params = spec.init_params(rng) if not hasattr(self, "_params") else self._params
        if not hasattr(self, "_params"):
            self._params = params
        self._params[:n_shared] = global_shared
        idx = np.arange(len(self.Xtr))
        for _ in range(LOCAL_EPOCHS):
            rng.shuffle(idx)
            train_epoch(self._params, self.Xtr[idx], self.ytr[idx], LR, BATCH, self.pos_weight, rng)
        score = evaluate(self._params, self.Xte, self.yte)["balanced_accuracy"]
        return self._params[:n_shared].copy(), len(self.Xtr), score


def pooled_scaling(clients):
    """The ONE set of scaling constants every node must use: mean/std of the TRAIN rows of all
    usable clients pooled together (never test rows). Deterministic for a given CSV, so the
    simulated patients, the ESP32 export and the model export all get identical numbers."""
    return zscore_fit(np.concatenate([c.Xtr_raw for c in clients]))


def pick_clients(records, y, X):
    uniq = sorted(set(records))
    clients = []
    excluded = []
    for rec in uniq:
        mask = records == rec
        n = mask.sum()
        n_pos = int(y[mask].sum())
        if n < MIN_TEST_SIZE or n_pos < MIN_TEST_POS:
            excluded.append((rec, n, n_pos))
            continue
        clients.append(Client(rec, X[mask], y[mask]))
    return clients, excluded


def run_experiment(clients, mode, seed=0):
    n_shared = spec.N_BASE if mode == "fedper" else spec.N_TOTAL
    rng = np.random.default_rng(seed)
    global_shared = spec.init_params(rng)[:n_shared].copy()
    history = []
    for r in range(ROUNDS):
        updates = []
        scores = []
        for c in clients:
            shared, n_train, score = c.local_round(global_shared, n_shared, rng)
            updates.append((shared, n_train))
            scores.append(score)
        ns = np.array([u[1] for u in updates], dtype=np.float64)
        W = np.stack([u[0] for u in updates])
        global_shared = (W * (ns / ns.sum())[:, None]).sum(axis=0).astype(np.float32)
        history.append(dict(round=r, mean_score=float(np.mean(scores)),
                            std_score=float(np.std(scores)), per_client=dict(zip(
                                [c.record for c in clients], scores))))
    return history, n_shared


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "apnea_ecg_features.csv"
    records, y, X = load_csv(path)
    clients, excluded = pick_clients(records, y, X)
    print(f"{len(clients)} usable clients: {[c.record for c in clients]}")
    if excluded:
        print(f"{len(excluded)} excluded (too few rows/positives): "
              f"{[(r, n, p) for r, n, p in excluded]}")
    if len(clients) < 2:
        print("need at least 2 clients to compare federation - check your CSV.")
        return

    # fixed normalization constants, pooled TRAIN data only, never test data
    all_train = np.concatenate([c.Xtr_raw for c in clients])
    mu, sd = zscore_fit(all_train)
    for c in clients:
        c.apply_norm(mu, sd)
    print(f"normalization fit on {len(all_train)} pooled training rows across all clients\n")

    results = {}
    for mode in ["fedavg", "fedper"]:
        for c in clients:
            if hasattr(c, "_params"):
                del c._params
        print(f"=== {mode.upper()} ===")
        history, n_shared = run_experiment(clients, mode)
        bytes_down = 10 + 4 * n_shared
        bytes_up = 18 + 4 * n_shared
        for h in history:
            if h["round"] % 5 == 0 or h["round"] == ROUNDS - 1:
                print(f"  round {h['round']:2d}: mean balanced_acc={h['mean_score']:.3f} "
                      f"+/- {h['std_score']:.3f}")
        final = history[-1]
        print(f"  final: mean={final['mean_score']:.3f} +/- {final['std_score']:.3f}  "
              f"per-client: { {k: round(v,3) for k,v in final['per_client'].items()} }")
        print(f"  bytes/round/client: {bytes_down} down, {bytes_up} up "
              f"({n_shared} shared params)\n")
        results[mode] = (history, bytes_down, bytes_up)

    fa_final = results["fedavg"][0][-1]["mean_score"]
    fp_final = results["fedper"][0][-1]["mean_score"]
    print("=== SUMMARY (single seed) ===")
    print(f"  FedAvg final mean balanced_acc: {fa_final:.3f}  "
          f"({results['fedavg'][2]} bytes/round/client up)")
    print(f"  FedPer final mean balanced_acc: {fp_final:.3f}  "
          f"({results['fedper'][2]} bytes/round/client up)")
    savings = 100 * (1 - results['fedper'][2] / results['fedavg'][2])
    print(f"  FedPer moves {savings:.1f}% fewer bytes/round than FedAvg "
          f"(always true by construction: {spec.N_BASE} vs {spec.N_TOTAL} shared params)")
    print(f"  accuracy difference (FedPer - FedAvg): {fp_final - fa_final:+.3f}")

    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"\n=== MULTI-SEED CHECK ({n_seeds} seeds, same data/clients, different "
          f"init + local shuffling) ===")
    fa_finals, fp_finals = [], []
    for seed in range(n_seeds):
        for c in clients:
            if hasattr(c, "_params"):
                del c._params
        h_fa, _ = run_experiment(clients, "fedavg", seed=seed)
        for c in clients:
            if hasattr(c, "_params"):
                del c._params
        h_fp, _ = run_experiment(clients, "fedper", seed=seed)
        fa_finals.append(h_fa[-1]["mean_score"])
        fp_finals.append(h_fp[-1]["mean_score"])
        print(f"  seed {seed}: FedAvg={fa_finals[-1]:.3f}  FedPer={fp_finals[-1]:.3f}  "
              f"diff={fp_finals[-1]-fa_finals[-1]:+.3f}")
    fa_finals = np.array(fa_finals)
    fp_finals = np.array(fp_finals)
    print(f"\n  FedAvg across seeds: {fa_finals.mean():.3f} +/- {fa_finals.std():.3f}")
    print(f"  FedPer across seeds: {fp_finals.mean():.3f} +/- {fp_finals.std():.3f}")
    diffs = fp_finals - fa_finals
    print(f"  FedPer-FedAvg diff: {diffs.mean():+.3f} +/- {diffs.std():.3f}  "
          f"({'FedPer consistently behind' if (diffs < 0).all() else 'FedAvg consistently behind' if (diffs > 0).all() else 'sign is NOT consistent across seeds - treat the single-seed result with caution'})")


if __name__ == "__main__":
    main()
