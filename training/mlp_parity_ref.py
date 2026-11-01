#!/usr/bin/env python3
"""
Reference implementation for the ESP32 MLP parity test (esp32_mlp_test.ino).

Everything is deterministic and float32, so the ESP32 must reproduce these
numbers (tiny float rounding differences are expected, ~1e-6).

    python mlp_parity_ref.py

Same model as spec.py: 8 -> 16 (ReLU) -> 8 (ReLU) -> 1 (sigmoid), 289 params.
"""

import numpy as np

F = np.float32
IN = [8, 16, 8]
OUT = [16, 8, 1]
N_TOTAL = sum(i * o + o for i, o in zip(IN, OUT))   # 289

LR, BATCH, EPOCHS, N_DATA = F(0.05), 16, 5, 64


class Lcg:
    """32-bit LCG, identical on the ESP32 (no numpy RNG, which the board cannot reproduce)."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return F(self.s >> 8) * F(1.0 / 16777216.0)


def init_params(lcg):
    parts = []
    for i, o in zip(IN, OUT):
        a = np.sqrt(F(6.0) / F(i)).astype(F)
        w = np.array([(lcg.uniform() * F(2.0) - F(1.0)) * a for _ in range(o * i)], dtype=F)
        parts.append(w)
        parts.append(np.zeros(o, dtype=F))
    return np.concatenate(parts).astype(F)


def make_data():
    lcg = Lcg(12345)
    X = np.zeros((N_DATA, 8), dtype=F)
    y = np.zeros(N_DATA, dtype=F)
    for n in range(N_DATA):
        for j in range(8):
            X[n, j] = lcg.uniform() * F(4.0) - F(2.0)
        s = X[n, 0] + F(0.5) * X[n, 1] - F(0.5) * X[n, 2] + F(0.3) * X[n, 3]
        flip = lcg.uniform()
        label = 1 if s > F(0.2) else 0
        if flip < F(0.1):
            label = 1 - label
        y[n] = F(label)
    return X, y


def unpack(P):
    out, pos = [], 0
    for i, o in zip(IN, OUT):
        W = P[pos:pos + o * i].reshape(o, i); pos += o * i
        b = P[pos:pos + o]; pos += o
        out.append((W, b))
    return out


def forward(P, X):
    layers = unpack(P)
    acts = [X]
    a = X
    for k, (W, b) in enumerate(layers):
        z = a @ W.T + b
        a = np.maximum(z, F(0)) if k < 2 else (F(1) / (F(1) + np.exp(-z))).astype(F)
        acts.append(a.astype(F))
    return acts


def loss(P, X, y, pw):
    p = forward(P, X)[-1].reshape(-1).astype(np.float64)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yy = y.astype(np.float64)
    return float(np.mean(-(float(pw) * yy * np.log(p) + (1 - yy) * np.log(1 - p))))


def train_epoch(P, X, y, pw):
    layers = unpack(P)
    for s in range(0, len(X), BATCH):
        xb, yb = X[s:s + BATCH], y[s:s + BATCH].reshape(-1, 1)
        acts = forward(P, xb)
        p = acts[-1]
        dz = (p * (F(1) - yb) - pw * yb * (F(1) - p)).astype(F)
        for l in range(2, -1, -1):
            W, b = layers[l]
            gW = (dz.T @ acts[l]).astype(F)
            gb = dz.sum(axis=0).astype(F)
            if l > 0:
                dz_next = ((dz @ W) * (acts[l] > 0)).astype(F)
            scale = F(LR / F(len(xb)))
            W -= scale * gW
            b -= scale * gb
            if l > 0:
                dz = dz_next


def main():
    X, y = make_data()
    P = init_params(Lcg(777))
    n_pos = float(y.sum())
    pw = F((N_DATA - n_pos) / n_pos)
    print(f"params={len(P)} samples={N_DATA} pos={int(n_pos)} pos_weight={float(pw):.6f}")
    print(f"loss[0] = {loss(P, X, y, pw):.6f}")
    for e in range(1, EPOCHS + 1):
        train_epoch(P, X, y, pw)
        print(f"loss[{e}] = {loss(P, X, y, pw):.6f}")
    print(f"sum = {float(P.astype(np.float64).sum()):.6f}")
    print(f"sumabs = {float(np.abs(P).astype(np.float64).sum()):.6f}")
    print("P[0..3] = " + " ".join(f"{float(v):.6f}" for v in P[:4]))


if __name__ == "__main__":
    main()
