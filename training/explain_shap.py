#!/usr/bin/env python3
"""
Explainability for the frozen model (rundown Stage 11, gap 5, metric 5).

    python explain_shap.py apnea_ecg_features.csv --model esp32_model_validate/model_export.h

Explains the model that is actually deployed (the weights and scaling in model_export.h) on every
late-night test window of every patient (the same windows the model was scored on).

Method: EXACT interventional Shapley values. The model has only 8 inputs, so instead of sampling
(KernelSHAP) every one of the 2^8 = 256 feature coalitions is evaluated. For a coalition S, the features in
S keep the explained window's values and the others are drawn from a background of real TRAIN windows,
averaged over the background. This is the quantity the `shap` library's KernelExplainer approximates; here it
is computed exactly. Values are in PROBABILITY units: for every window they add up to
(model probability for that window) - (average model probability over the background).

Outputs (in --out, default shap_results/): shap_summary.md, shap_values.npz and five figures.
Optional: --crosscheck compares a few windows against the `shap` library (pip install shap).
"""
import argparse
import os
import re

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_apply
from fedavg_vs_fedper_real import pick_clients, pooled_scaling


# ------------------------------------------------------------------ the model (batch forward, float32)
def parse_header(path):
    s = open(path).read()

    def arr(name):
        m = re.search(rf"static const float {name}\[\d+\] = \{{(.*?)\}};", s, re.S)
        if not m:
            raise SystemExit(f"{path}: cannot find {name} - is this a model_export.h?")
        return np.array([float(t.strip().rstrip("f")) for t in m.group(1).replace("\n", " ").split(",")
                         if t.strip()], dtype=np.float32)
    return arr("MODEL_W"), arr("FEAT_MU"), arr("FEAT_SD")


class Net:
    def __init__(self, w):
        self.W1, self.b1 = w[0:128].reshape(16, 8), w[128:144]
        self.W2, self.b2 = w[144:272].reshape(8, 16), w[272:280]
        self.W3, self.b3 = w[280:288].reshape(1, 8), w[288:289]

    def __call__(self, X):                       # X: (n, 8) already scaled -> probability (n,)
        X = np.asarray(X, dtype=np.float32)
        a = np.maximum(X @ self.W1.T + self.b1, 0)
        a = np.maximum(a @ self.W2.T + self.b2, 0)
        z = a @ self.W3.T + self.b3
        return (1.0 / (1.0 + np.exp(-z))).reshape(-1)


# ------------------------------------------------------------------ exact Shapley values
def shapley_weights(n):
    """(n, 2^n) matrix Wsh with  phi = Wsh @ v,  v[S] = value of coalition S (bit i of S = feature i is 'present')."""
    from math import factorial
    W = np.zeros((n, 2 ** n))
    for i in range(n):
        for S in range(2 ** n):
            if S & (1 << i):
                continue
            k = bin(S).count("1")
            c = factorial(k) * factorial(n - k - 1) / factorial(n)
            W[i, S | (1 << i)] += c
            W[i, S] -= c
    return W


def exact_shapley(f, X, bg, chunk=16):
    """Exact interventional Shapley values of f at each row of X against background rows bg.
    Returns (phi (m, n), base (scalar = mean f over bg), fx (m,))."""
    m, n = X.shape
    masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)      # (2^n, n)
    Wsh = shapley_weights(n)
    phi = np.zeros((m, n))
    fx = np.zeros(m)
    for s in range(0, m, chunk):
        xs = X[s:s + chunk]
        k = len(xs)
        mix = np.where(masks[None, :, None, :], xs[:, None, None, :], bg[None, None, :, :])   # (k, 2^n, B, n)
        v = f(mix.reshape(-1, n)).reshape(k, 2 ** n, len(bg)).mean(axis=2)                    # (k, 2^n)
        phi[s:s + k] = v @ Wsh.T
        fx[s:s + k] = v[:, -1]                                                                 # all features present = f(x)
    base = float(f(bg).mean())
    return phi, base, fx


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="apnea_ecg_features.csv")
    ap.add_argument("--model", default="esp32_model_validate/model_export.h")
    ap.add_argument("--out", default="shap_results")
    ap.add_argument("--background", type=int, default=100, help="number of TRAIN windows used as background")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crosscheck", action="store_true", help="compare a few windows with the shap library")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    w, mu_h, sd_h = parse_header(a.model)
    net = Net(w)
    names = list(spec.FEATURES)

    records, y, X = load_csv(a.csv)
    clients, _ = pick_clients(records, y, X)
    mu, sd = pooled_scaling(clients)
    same = np.allclose(mu, mu_h, atol=1e-5) and np.allclose(sd, sd_h, atol=1e-5)
    print(f"{len(clients)} patients; scaling in the model file {'matches' if same else 'DOES NOT MATCH'} "
          f"the one computed from this CSV (largest difference {max(np.abs(mu - mu_h).max(), np.abs(sd - sd_h).max()):.1e})")
    if not same:
        print("  warning: this CSV is not the one the model was trained on; the explanation is still of the model, "
              "but its 'background' differs from its training data.")

    rng = np.random.default_rng(a.seed)
    train_raw = np.concatenate([c.Xtr_raw for c in clients])
    pick = rng.choice(len(train_raw), size=min(a.background, len(train_raw)), replace=False)
    bg = zscore_apply(train_raw[pick], mu_h, sd_h).astype(np.float32)             # the model's own scaling

    Xte_raw = np.concatenate([c.Xte_raw for c in clients])
    yte = np.concatenate([c.yte for c in clients]).astype(int)
    pid = np.concatenate([[k] * len(c.yte) for k, c in enumerate(clients)])
    pnames = [str(c.record) for c in clients]
    Xs = zscore_apply(Xte_raw, mu_h, sd_h).astype(np.float32)
    print(f"explaining {len(Xs)} test windows against a background of {len(bg)} train windows "
          f"(2^8 = 256 coalitions each, exact) ...", flush=True)
    phi, base, fx = exact_shapley(net, Xs, bg)
    prob = net(Xs)
    eff = np.abs(phi.sum(1) + base - prob).max()
    print(f"check: the 8 contributions add up to (probability - average) for every window; largest error {eff:.1e}")

    mean_abs = np.abs(phi).mean(0)
    order = np.argsort(-mean_abs)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"## Global importance (mean |contribution| in probability points, {len(Xs)} test windows, {len(clients)} patients)\n")
    out("| rank | feature | mean abs contribution | share of total | effect of a HIGHER value (rank correlation of value vs contribution) |")
    out("|---|---|---|---|---|")
    for r, i in enumerate(order, 1):
        rho = spearman(Xte_raw[:, i], phi[:, i])
        way = "raises the apnea probability" if rho > 0.3 else ("lowers it" if rho < -0.3 else "mixed / weak")
        out(f"| {r} | {names[i]} | {mean_abs[i]:.4f} | {100 * mean_abs[i] / mean_abs.sum():.0f}% | {rho:+.2f} ({way}) |")
    out(f"\nAverage model probability over the background: {base:.3f}; over the test windows: {prob.mean():.3f}.\n")

    out("## Which features separate real apnea windows from normal ones? (mean contribution, probability points)\n")
    out("| feature | true apnea windows | true normal windows | difference |")
    out("|---|---|---|---|")
    for i in order:
        pa, pn = phi[yte == 1, i].mean(), phi[yte == 0, i].mean()
        out(f"| {names[i]} | {pa:+.4f} | {pn:+.4f} | {pa - pn:+.4f} |")
    out(f"\nModel probability: true apnea windows {prob[yte == 1].mean():.3f} vs true normal windows {prob[yte == 0].mean():.3f}.\n")

    out("## Top 3 features per patient (mean |contribution|)\n")
    out("| patient | windows | 1st | 2nd | 3rd |")
    out("|---|---|---|---|---|")
    per = np.zeros((len(clients), 8))
    for k in range(len(clients)):
        per[k] = np.abs(phi[pid == k]).mean(0)
        top = np.argsort(-per[k])[:3]
        out(f"| {pnames[k]} | {int((pid == k).sum())} | " + " | ".join(f"{names[i]} ({per[k][i]:.3f})" for i in top) + " |")
    out()

    if a.crosscheck:
        try:
            import shap
            sub = np.arange(0, len(Xs), max(len(Xs) // 12, 1))[:12]
            ex = shap.KernelExplainer(lambda z: net(np.asarray(z, dtype=np.float32)), bg)
            ref = np.asarray(ex.shap_values(Xs[sub], nsamples=2 ** 8 - 2))
            ref = ref.reshape(len(sub), 8)
            out(f"## Cross-check with the shap library (KernelExplainer, all coalitions) on {len(sub)} windows\n")
            out(f"- largest difference from the exact values above: {np.abs(ref - phi[sub]).max():.2e} probability points\n")
        except ImportError:
            out("## Cross-check skipped: the shap library is not installed (pip install shap)\n")
        except Exception as e:
            out(f"## Cross-check with the shap library failed: {type(e).__name__}: {e}\n")

    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "shap_values.npz"), phi=phi, base=base, prob=prob, y=yte, patient=pid,
             patient_names=np.array(pnames), features=np.array(names), X_raw=Xte_raw)
    with open(os.path.join(a.out, "shap_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {a.out}/shap_summary.md and {a.out}/shap_values.npz")

    if a.no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so no figures: pip install matplotlib")
        return

    # 1. global importance
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="#4c72b0")
    ax.set_xlabel("mean |contribution| (probability points)"); ax.set_title("Which features drive the model?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shap_importance.png"), dpi=160); plt.close(fig)

    # 2. beeswarm: every window, coloured by the feature's value
    fig, ax = plt.subplots(figsize=(8, 4.8))
    jr = np.random.default_rng(1)
    for row, i in enumerate(order[::-1]):
        rank = np.argsort(np.argsort(Xte_raw[:, i])) / max(len(Xte_raw) - 1, 1)
        ax.scatter(phi[:, i], row + jr.uniform(-0.3, 0.3, len(phi)), c=rank, cmap="coolwarm", s=6, alpha=0.7, linewidths=0)
    ax.set_yticks(range(8)); ax.set_yticklabels([names[i] for i in order[::-1]]); ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("contribution to the apnea probability (probability points)")
    ax.set_title("Every test window: red = high feature value, blue = low")
    sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"])
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shap_beeswarm.png"), dpi=160); plt.close(fig)

    # 3. waterfalls for four example windows
    cand = {}
    ok_idx = np.arange(len(prob))
    for label, mask, pick_hi in (("true apnea, caught", (yte == 1) & (prob >= 0.5), True),
                                 ("true normal, correctly cleared", (yte == 0) & (prob < 0.5), False),
                                 ("normal window, FALSE ALARM", (yte == 0) & (prob >= 0.5), True),
                                 ("apnea window, MISSED", (yte == 1) & (prob < 0.5), False)):
        idx = ok_idx[mask]
        if len(idx):
            cand[label] = idx[np.argmax(prob[idx])] if pick_hi else idx[np.argmin(prob[idx])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (label, i) in zip(axes.ravel(), cand.items()):
        o = np.argsort(-np.abs(phi[i]))[::-1]
        cum = base
        for r, j in enumerate(o):
            ax.barh(r, phi[i, j], left=cum, color="#d95f02" if phi[i, j] > 0 else "#1b9e77")
            cum += phi[i, j]
        ax.set_yticks(range(8)); ax.set_yticklabels([f"{names[j]} = {Xte_raw[i, j]:.3g}" for j in o], fontsize=8)
        ax.axvline(base, color="grey", lw=0.8, ls=":"); ax.axvline(prob[i], color="k", lw=0.8)
        ax.set_title(f"{label}\npatient {pnames[pid[i]]}, probability {prob[i]:.2f} (average {base:.2f})", fontsize=9)
        ax.set_xlabel("probability"); ax.grid(axis="x", alpha=0.3)
    for ax in axes.ravel()[len(cand):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shap_waterfall_examples.png"), dpi=160); plt.close(fig)

    # 4. who is explained by what: per-patient heatmap
    fig, ax = plt.subplots(figsize=(8, 4.3))
    im = ax.imshow(per[:, order], cmap="viridis", aspect="auto")
    ax.set_xticks(range(8)); ax.set_xticklabels([names[i] for i in order], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pnames))); ax.set_yticklabels(pnames)
    for r in range(len(pnames)):
        for c in range(8):
            ax.text(c, r, f"{per[r, order[c]]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, label="mean |contribution|"); ax.set_title("Feature importance differs between patients")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shap_by_patient.png"), dpi=160); plt.close(fig)

    # 5. dependence of the top two features
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, i in zip(axes, order[:2]):
        ax.scatter(Xte_raw[:, i], phi[:, i], c=np.where(yte == 1, "#d95f02", "#4c72b0"), s=6, alpha=0.6, linewidths=0)
        ax.axhline(0, color="k", lw=0.7); ax.set_xlabel(names[i]); ax.set_ylabel("contribution (probability points)")
        ax.grid(alpha=0.3)
    axes[0].set_title("orange = true apnea window, blue = true normal window", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shap_dependence.png"), dpi=160); plt.close(fig)
    print(f"saved 5 figures in {a.out}/")


if __name__ == "__main__":
    main()
