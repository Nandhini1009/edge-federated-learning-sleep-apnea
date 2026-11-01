#!/usr/bin/env python3
"""
Centralized vs FedAvg vs FedPer on the real PhysioNet patients (rundown Stage 9 and metrics 1, 2, 4).

    python run_experiments.py apnea_ecg_features.csv                 # 100 rounds x 5 seeds
    python run_experiments.py apnea_ecg_features.csv --quick         # 30 rounds x 3 seeds, a fast check

It reuses the project's own pieces unchanged (same clients, same time-blocked 70/30 split per patient,
same shared scaling, same local training: lr 0.05, batch 32, 3 epochs per round, sample-weighted
aggregation), so the numbers stay comparable with fedavg_vs_fedper_real.py.

Methods
  Centralized : ONE model trained on all patients' training rows pooled together (no privacy).
                One "round" = 3 epochs over the pooled data. Uses ONE pooled apnea weight.
  Centralized (per-patient weights) : the same, but every patient's apnea windows are weighted by
                that patient's OWN ratio, exactly as each federated patient does in its local loss.
                This is the like-for-like upper bound; the first one is kept for comparison.
  FedAvg      : the whole 289-weight model is shared and averaged every round.
  FedPer      : only the 280-weight base is shared; each patient keeps a private 9-weight head.

Two ways of scoring a federated method, because they answer different questions
  deployed   : the model a patient would actually USE after a round: FedAvg = the global model,
               FedPer = global base + that patient's own head. This is the number for the paper.
  post-local : the patient's model right after its 3 local epochs (what fedavg_vs_fedper_real.py
               reported, and what the ESP32 printed). Includes a little local fine-tuning.

Every score is on that patient's held-out LATE-night windows. Balanced accuracy uses the 0.5 cutoff;
AUC does not depend on any cutoff. "Calibrated" balanced accuracy lets each patient pick its OWN cutoff
using only its own EARLY (training) windows - what a deployed patient could do after a calibration
period - and then scores it on the late windows; no test window is ever used to pick a cutoff. Writes: results/experiments.npz, results/results_table.md and
three figures (needs matplotlib for the figures: pip install matplotlib).
"""
import argparse
import os
import time

import numpy as np

import spec
from train_on_real_apnea import load_csv, forward, train_epoch, evaluate
from fedavg_vs_fedper_real import pick_clients, pooled_scaling, LOCAL_EPOCHS, LR, BATCH


def auc_score(y, p):
    """ROC AUC (Mann-Whitney, ties averaged). NaN if a patient has only one class in the test windows."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


THRESHOLDS = np.linspace(0.05, 0.95, 19)


def bal_at(p, y, thresholds):
    """Balanced accuracy of probabilities p against labels y at each threshold (vectorised)."""
    pred = p[:, None] >= thresholds[None, :]
    pos, neg = (y == 1), (y == 0)
    tpr = (pred & pos[:, None]).sum(0) / max(pos.sum(), 1)
    tnr = ((~pred) & neg[:, None]).sum(0) / max(neg.sum(), 1)
    return 0.5 * (tpr + tnr)


def pick_threshold(p_train, y_train):
    """The cutoff that is best on the patient's own EARLY windows (ties -> nearest 0.5). Never sees late windows."""
    tb = bal_at(p_train, y_train, THRESHOLDS)
    return float(THRESHOLDS[np.argmax(tb - 1e-9 * np.abs(THRESHOLDS - 0.5))])


def calibrated_bal(p_train, y_train, p_test, y_test):
    """Score the late windows at the cutoff chosen from the early windows."""
    return float(bal_at(p_test, y_test, np.array([pick_threshold(p_train, y_train)]))[0])


def score(params, c):
    """(balanced accuracy at 0.5, AUC, calibrated balanced accuracy) on one client's late windows."""
    p_te = forward(params, c.Xte).reshape(-1)
    bal = evaluate(params, c.Xte, c.yte)["balanced_accuracy"]
    cal = calibrated_bal(forward(params, c.Xtr).reshape(-1), c.ytr, p_te, c.yte)
    return bal, auc_score(c.yte, p_te), cal


def train_epoch_weighted(params, X, y, w_pos, lr, batch, rng):
    """train_epoch from train_on_real_apnea.py, but with a per-window apnea weight (w_pos[i] for window i).
    With a constant w_pos it is identical to train_epoch (checked in the tests)."""
    layers = spec.unpack(params)
    idx = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        bi = idx[s:s + batch]
        yb = y[bi].reshape(-1, 1)
        wb = w_pos[bi].reshape(-1, 1).astype(yb.dtype)       # same precision as the scalar weight in train_epoch
        acts = [X[bi]]
        a = X[bi]
        for i, (W, b) in enumerate(layers):
            z = a @ W.T + b
            a = np.maximum(z, 0.0) if i < len(layers) - 1 else 1.0 / (1.0 + np.exp(-z))
            acts.append(a)
        p = acts[-1]
        dz = (p * (1.0 - yb) - wb * yb * (1.0 - p)) / len(bi)
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


def run_federated(clients, mode, seed, rounds):
    n_shared = spec.N_BASE if mode == "fedper" else spec.N_TOTAL
    for c in clients:
        if hasattr(c, "_params"):
            del c._params
    rng = np.random.default_rng(seed)
    global_shared = spec.init_params(rng)[:n_shared].copy()
    K = len(clients)
    dep_bal, dep_auc, dep_cal, loc_bal = (np.zeros((rounds, K)) for _ in range(4))
    for r in range(rounds):
        updates = []
        for k, c in enumerate(clients):
            shared, n_train, local_score = c.local_round(global_shared, n_shared, rng)
            updates.append((shared, n_train))
            loc_bal[r, k] = local_score
        ns = np.array([u[1] for u in updates], dtype=np.float64)
        W = np.stack([u[0] for u in updates])
        global_shared = (W * (ns / ns.sum())[:, None]).sum(axis=0).astype(np.float32)
        for k, c in enumerate(clients):
            p = global_shared if mode == "fedavg" else np.concatenate([global_shared, c._params[spec.N_BASE:]])
            dep_bal[r, k], dep_auc[r, k], dep_cal[r, k] = score(p, c)
    return dep_bal, dep_auc, dep_cal, loc_bal


def run_centralized(clients, seed, rounds, per_patient=False):
    rng = np.random.default_rng(seed)
    params = spec.init_params(rng).copy()
    Xtr = np.concatenate([c.Xtr for c in clients])
    ytr = np.concatenate([c.ytr for c in clients])
    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    w_pos = np.concatenate([np.full(len(c.ytr), c.pos_weight) for c in clients])   # each patient's own ratio
    idx = np.arange(len(Xtr))
    K = len(clients)
    bal, auc, cal = (np.zeros((rounds, K)) for _ in range(3))
    for r in range(rounds):
        for _ in range(LOCAL_EPOCHS):
            rng.shuffle(idx)
            if per_patient:
                train_epoch_weighted(params, Xtr[idx], ytr[idx], w_pos[idx], LR, BATCH, rng)
            else:
                train_epoch(params, Xtr[idx], ytr[idx], LR, BATCH, pos_weight, rng)
        for k, c in enumerate(clients):
            bal[r, k], auc[r, k], cal[r, k] = score(params, c)
    return bal, auc, cal


def pm(a):
    return f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="apnea_ecg_features.csv")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tail", type=int, default=10, help="average the last N rounds (single rounds are noisy)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="30 rounds x 3 seeds")
    ap.add_argument("--min-class", type=int, default=15,
                    help="a patient is scored per-patient only if EVERY class has at least this many windows in both the early and the late part of the night")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.rounds, a.seeds = 30, 3
    a.tail = min(a.tail, a.rounds)

    records, y, X = load_csv(a.csv)
    clients, excluded = pick_clients(records, y, X)
    names = [str(c.record) for c in clients]
    enough = np.array([min(int(c.ytr.sum()), len(c.ytr) - int(c.ytr.sum()), int(c.yte.sum()), len(c.yte) - int(c.yte.sum())) >= a.min_class
                       for c in clients])
    disp = [n if e else n + "*" for n, e in zip(names, enough)]        # '*' = too few windows in one class
    EN = np.where(enough)[0]
    if len(clients) < 2:
        raise SystemExit("need at least 2 usable patients - check the CSV")
    mu, sd = pooled_scaling(clients)
    for c in clients:
        c.apply_norm(mu, sd)
    print(f"{len(clients)} patients {names}; shared scaling; {a.rounds} rounds x {a.seeds} seeds; "
          f"scores = last {a.tail} rounds averaged\n", flush=True)

    S, R, K = a.seeds, a.rounds, len(clients)
    METHODS = ("central", "central_pw", "fedavg", "fedper")
    res = {m: dict(bal=np.zeros((S, R, K)), auc=np.zeros((S, R, K)), cal=np.zeros((S, R, K))) for m in METHODS}
    loc = {m: np.zeros((S, R, K)) for m in ("fedavg", "fedper")}
    t0 = time.time()
    for s in range(S):
        for m, pp in (("central", False), ("central_pw", True)):
            res[m]["bal"][s], res[m]["auc"][s], res[m]["cal"][s] = run_centralized(clients, s, R, per_patient=pp)
        for m in ("fedavg", "fedper"):
            res[m]["bal"][s], res[m]["auc"][s], res[m]["cal"][s], loc[m][s] = run_federated(clients, m, s, R)
        print(f"  seed {s} done ({time.time() - t0:.0f} s)", flush=True)

    def tail_mean(arr):                      # (S, R, K) -> (S, K): average of the last rounds
        return arr[:, -a.tail:, :].mean(axis=1)

    T = {m: tail_mean(res[m]["bal"]) for m in res}
    TA = {m: tail_mean(res[m]["auc"]) for m in res}
    TC = {m: tail_mean(res[m]["cal"]) for m in res}
    TL = {m: tail_mean(loc[m]) for m in loc}
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    H = "| patient | Centralized | Centralized (per-patient weights) | FedAvg | FedPer |\n|---|---|---|---|---|"

    def table(title, TT, nan=False):
        out(title + "\n")
        out(H)
        for k, nme in enumerate(disp):
            out(f"| {nme} | " + " | ".join(pm(TT[m][:, k]) for m in METHODS) + " |")
        agg = (lambda a: np.nanmean(a, 1))
        out("| **mean of all patients** | " + " | ".join(f"**{pm(agg(TT[m]))}**" for m in METHODS) + " |")
        if len(EN):
            out(f"| **mean of the {len(EN)} patients with enough windows** | " + " | ".join(f"**{pm(agg(TT[m][:, EN]))}**" for m in METHODS) + " |")
        if nan:
            out("| median of patients | " + " | ".join(f"{np.nanmedian(TT[m].mean(0)):.3f}" for m in METHODS) + " |")
            out("| patients above 0.5 | " + " | ".join(f"{int((np.nanmean(TT[m], 0) > 0.5).sum())} of {K}" for m in METHODS) + " |")
        out()

    out(f"## Test windows per patient (the late 30% of the night): how many are apnea / normal\n")
    out("| patient | train windows (apnea) | test windows (apnea / normal) | enough windows? |")
    out("|---|---|---|---|")
    for c, nme, e in zip(clients, names, enough):
        npos = int(c.yte.sum())
        out(f"| {nme} | {len(c.ytr)} ({int(c.ytr.sum())}) | {len(c.yte)} ({npos} / {len(c.yte) - npos}) | {'yes' if e else 'NO (*)'} |")
    out(f"\n(*) = fewer than {a.min_class} windows of some class in the early or the late part of the night. Per-patient scores for these "
        f"patients are mostly noise; they still take part in training. {len(EN)} of {K} patients have enough windows.\n")

    table(f"## Balanced accuracy at the 0.5 cutoff - deployed model, mean of the last {a.tail} of {R} rounds, {S} seeds (mean ± std over seeds)", T)
    table("## AUC (does the model separate apnea from normal at all? 0.5 = no) - same setting", TA, nan=True)
    table("## Balanced accuracy after each patient picks its OWN cutoff from its EARLY windows (calibration)", TC)

    d = T["fedper"].mean(1) - T["fedavg"].mean(1)
    dk = (T["fedper"] - T["fedavg"]).mean(0)
    out("\n## Is personalization (FedPer) better than FedAvg?\n")
    out(f"- mean difference FedPer - FedAvg, deployed: {d.mean():+.3f} ± {d.std():.3f} over seeds "
        f"({int((d > 0).sum())} of {S} seeds favour FedPer)")
    out(f"- patients where FedPer is better (averaged over seeds): {int((dk > 0).sum())} of {K}: "
        + ", ".join(f"{n} {v:+.3f}" for n, v in zip(names, dk)))
    dl = TL["fedper"].mean(1) - TL["fedavg"].mean(1)
    if len(EN):
        de = (T["fedper"][:, EN] - T["fedavg"][:, EN]).mean(1)
        dae = np.nanmean(TA["fedper"][:, EN] - TA["fedavg"][:, EN], 1)
        out(f"- ONLY the {len(EN)} patients with enough windows ({', '.join(names[i] for i in EN)}): FedPer - FedAvg = {de.mean():+.3f} ± {de.std():.3f} "
            f"in balanced accuracy, {dae.mean():+.3f} ± {dae.std():.3f} in AUC; centralized (per-patient weights) - FedAvg = "
            f"{(T['central_pw'][:, EN] - T['fedavg'][:, EN]).mean():+.3f}")
    out(f"- same comparison right after local training (the older 'post-local' view): {dl.mean():+.3f} ± {dl.std():.3f}")
    out(f"- centralized (pooled weight) minus FedAvg: {(T['central'].mean(1) - T['fedavg'].mean(1)).mean():+.3f}")
    out(f"- centralized (per-patient weights, the like-for-like baseline) minus FedAvg: "
        f"{(T['central_pw'].mean(1) - T['fedavg'].mean(1)).mean():+.3f}  (the price of keeping the data on the devices)")
    dc = TC["fedper"].mean(1) - TC["fedavg"].mean(1)
    out(f"- FedPer - FedAvg AFTER per-patient cutoff calibration: {dc.mean():+.3f} ± {dc.std():.3f} over seeds; "
        f"patients where FedPer is better: " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(names, (TC['fedper'] - TC['fedavg']).mean(0))))
    da = np.nanmean(TA["fedper"] - TA["fedavg"], 1)
    out(f"- FedPer - FedAvg in AUC: {da.mean():+.3f} ± {da.std():.3f} over seeds; per patient: "
        + ", ".join(f"{n} {v:+.3f}" for n, v in zip(names, np.nanmean(TA['fedper'] - TA['fedavg'], 0))))
    out(f"- how much of that AUC gap comes from one patient? mean AUC difference without the biggest-gap patient: "
        f"{np.delete(np.nanmean(TA['fedper'] - TA['fedavg'], 0), int(np.nanargmax(np.abs(np.nanmean(TA['fedper'] - TA['fedavg'], 0))))).mean():+.3f}")

    out("\n## Run-to-run spread (std over seeds of the patient-mean score)\n")
    out(f"- Centralized {T['central'].mean(1).std():.3f} | Centralized (per-patient weights) {T['central_pw'].mean(1).std():.3f} | "
        f"FedAvg {T['fedavg'].mean(1).std():.3f} | FedPer {T['fedper'].mean(1).std():.3f}")

    out("\n## Communication per round per patient (exact protocol bytes)\n")
    out("| method | uploaded | downloaded | shared weights |")
    out("|---|---|---|---|")
    out(f"| FedAvg | {18 + 4 * spec.N_TOTAL} B | {10 + 4 * spec.N_TOTAL} B | {spec.N_TOTAL} |")
    out(f"| FedPer | {18 + 4 * spec.N_BASE} B | {10 + 4 * spec.N_BASE} B | {spec.N_BASE} |")
    out("| Centralized (either) | (the patient's data itself) | - | - |")

    if R >= 20:
        out("\n## Cross-check with the earlier experiment (post-local score, mean of patients, round 20)\n")
        out(f"- FedAvg {pm(loc['fedavg'][:, 19, :].mean(1))} | FedPer {pm(loc['fedper'][:, 19, :].mean(1))}  "
            f"(earlier: 0.664 ± 0.005 and 0.653 ± 0.018 over 5 seeds)")

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "experiments.npz"), names=np.array(names), rounds=R, seeds=S,
             **{f"{m}_{k}": res[m][k] for m in res for k in res[m]}, **{f"{m}_postlocal": loc[m] for m in loc})
    with open(os.path.join(a.out, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved {a.out}/experiments.npz and {a.out}/results_table.md")

    if a.no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so no figures: pip install matplotlib   (then run again)")
        return
    col = {"central": "#444444", "central_pw": "#8c8c8c", "fedavg": "#d95f02", "fedper": "#1b9e77"}
    lab = {"central": "Centralized (pooled weight)", "central_pw": "Centralized (per-patient weights)",
           "fedavg": "FedAvg", "fedper": "FedPer"}
    x = np.arange(1, R + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for m in METHODS:
        v = res[m]["bal"].mean(2)                       # (S, R) mean over patients
        ax[0].plot(x, v.mean(0), color=col[m], label=lab[m])
        ax[0].fill_between(x, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=col[m], alpha=0.15)
    for m in ("fedavg", "fedper"):
        v = loc[m].mean(2)
        ax[1].plot(x, v.mean(0), color=col[m], label=lab[m])
        ax[1].fill_between(x, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=col[m], alpha=0.15)
    ax[0].set_title("Deployed model (what a patient would use)")
    ax[1].set_title("Right after local training")
    for A in ax:
        A.set_xlabel("federated round"); A.grid(alpha=0.3); A.axhline(0.5, color="k", lw=0.6, ls=":")
    ax[0].set_ylabel("balanced accuracy")
    ax[0].legend(title=f"mean of {K} patients,\nband = ±1 std over {S} seeds", fontsize=8, title_fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_accuracy_over_rounds.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.2
    for i, m in enumerate(METHODS):
        ax.bar(np.arange(K) + (i - 1.5) * w, T[m].mean(0), w, yerr=T[m].std(0), color=col[m], label=lab[m], capsize=2)
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(disp); ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_ylabel("balanced accuracy"); ax.set_title(f"Per patient (last {a.tail} rounds, error bar = std over seeds)")
    ax.set_ylim(0, 1.18); ax.legend(fontsize=8, ncol=4, loc="upper center"); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(f"* fewer than {a.min_class} windows of a class (early or late part): that patient's score is mostly noise", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_per_patient.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, m in enumerate(METHODS):
        ax.bar(np.arange(K) + (i - 1.5) * w, TA[m].mean(0), w, yerr=TA[m].std(0), color=col[m], label=lab[m], capsize=2)
    ax.set_xticks(np.arange(K)); ax.set_xticklabels(disp); ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_ylabel("AUC"); ax.set_title(f"AUC per patient (0.5 = no separation; last {a.tail} rounds, error bar = std over seeds)")
    ax.set_ylim(0, 1.18); ax.legend(fontsize=8, ncol=4, loc="upper center"); ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(f"* fewer than {a.min_class} windows of a class (early or late part): that patient's score is mostly noise", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_auc_per_patient.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    diff = T["fedper"] - T["fedavg"]
    ax.bar(np.arange(K), diff.mean(0), yerr=diff.std(0), color=["#1b9e77" if v > 0 else "#d95f02" for v in diff.mean(0)], capsize=3)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(np.arange(K)); ax.set_xticklabels(disp)
    ax.set_ylabel("FedPer - FedAvg (balanced accuracy)"); ax.set_title("Who benefits from personalization?")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(f"* fewer than {a.min_class} windows of a class: mostly noise", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_fedper_minus_fedavg.png"), dpi=160); plt.close(fig)
    print(f"saved 4 figures in {a.out}/")


if __name__ == "__main__":
    main()
