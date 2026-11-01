#!/usr/bin/env python3
"""
Simulated FedApnea nodes (stand-ins for the ESP32s).

Each node: holds one synthetic "patient" (time-ordered 60 s HRV feature windows),
fetches the global shared weights, trains locally with SGD, evaluates on its
held-out last 30 %, and uploads only the shared part of its weights.

The wire protocol is identical to what the ESP32 firmware will use, so if this
works against fl_server.py, the ESP32 only has to reproduce the same bytes.

    python sim_nodes.py --nodes 5 --rounds 15

NOTE: the data here is SYNTHETIC. Scores only prove the plumbing works.
Do not quote them anywhere.
"""

import argparse
import json
import struct
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import spec

MODEL_HDR = struct.Struct("<4sIH")
UPDATE_HDR = struct.Struct("<4sIIfH")


# ------------------------------------------------------------------ data
def make_patient(idx, n_windows=480):
    """Synthetic patient: apnea comes in episodes; feature offsets differ per patient."""
    delta = np.random.default_rng(12345).normal(0.0, 1.0, spec.N_FEAT)
    delta = 1.4 * delta / np.linalg.norm(delta) * np.sqrt(spec.N_FEAT) / 2
    for attempt in range(50):
        rng = np.random.default_rng(1000 + idx * 97 + attempt)
        prevalence = rng.uniform(0.2, 0.55)
        p_10 = 0.08
        p_01 = prevalence * p_10 / (1 - prevalence)
        state = np.zeros(n_windows, dtype=np.int64)
        for t in range(1, n_windows):
            if state[t - 1] == 0:
                state[t] = 1 if rng.random() < p_01 else 0
            else:
                state[t] = 0 if rng.random() < p_10 else 1
        shift = rng.normal(0.0, 0.8, spec.N_FEAT)
        noise = rng.uniform(0.8, 1.2)
        X = shift + state[:, None] * delta + rng.normal(0.0, noise, (n_windows, spec.N_FEAT))
        cut = int(0.7 * n_windows)          # time-blocked split, no shuffling
        ytr, yte = state[:cut], state[cut:]
        if 0.1 < ytr.mean() < 0.9 and 0.1 < yte.mean() < 0.9:
            return (X[:cut].astype(np.float32), ytr.astype(np.float32),
                    X[cut:].astype(np.float32), yte.astype(np.float32))
    raise RuntimeError("could not build a usable patient")


# ------------------------------------------------------------------ model
def forward(params, X):
    layers = spec.unpack(params)
    acts = [X]
    a = X
    for i, (W, b) in enumerate(layers):
        z = a @ W.T + b
        a = np.maximum(z, 0.0) if i < len(layers) - 1 else 1.0 / (1.0 + np.exp(-z))
        acts.append(a)
    return acts


def train_epoch(params, X, y, lr, batch, pos_weight, rng):
    layers = spec.unpack(params)              # views: in-place updates change params
    idx = rng.permutation(len(X))
    for s in range(0, len(X), batch):
        bi = idx[s:s + batch]
        yb = y[bi].reshape(-1, 1)
        acts = forward(params, X[bi])
        p = acts[-1]
        dz = (p * (1.0 - yb) - pos_weight * yb * (1.0 - p)) / len(bi)   # weighted BCE
        for l in range(len(layers) - 1, -1, -1):
            W, b = layers[l]
            gW = dz.T @ acts[l]
            gb = dz.sum(axis=0)
            if l > 0:
                dz_next = (dz @ W) * (acts[l] > 0.0)   # use W before updating it
            W -= lr * gW
            b -= lr * gb
            if l > 0:
                dz = dz_next


def balanced_accuracy(params, X, y):
    pred = (forward(params, X)[-1].reshape(-1) >= 0.5).astype(np.float32)
    tpr = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    tnr = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float(0.5 * (tpr + tnr))


# ------------------------------------------------------------------ network
def fetch_model(server, node):
    with urllib.request.urlopen(f"{server}/model?node={node}", timeout=10) as r:
        body = r.read()
    magic, rnd, n = MODEL_HDR.unpack_from(body, 0)
    assert magic == b"FLM1", "bad magic"
    vec = np.frombuffer(body[MODEL_HDR.size:], dtype="<f4").astype(np.float32)
    assert len(vec) == n
    return rnd, vec


def fetch_round(server):
    with urllib.request.urlopen(f"{server}/round", timeout=10) as r:
        return json.loads(r.read())["round"]


def post_update(server, node, rnd, n_samples, score, shared):
    body = UPDATE_HDR.pack(b"FLU1", rnd, n_samples, score, len(shared)) \
        + shared.astype("<f4").tobytes()
    req = urllib.request.Request(f"{server}/update?node={node}", data=body, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def run_node(idx, server, rounds, epochs, lr, log):
    node = f"sim-{idx:02d}"
    Xtr, ytr, Xte, yte = make_patient(idx)
    rng = np.random.default_rng(idx)
    params = spec.init_params(rng)            # this node's own head lives here
    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    last = -1
    while True:
        if last >= 0:                          # already uploaded for round `last`:
            while fetch_round(server) == last:  # poll cheaply until the round closes
                time.sleep(0.05)
        rnd, shared = fetch_model(server, node)
        if rnd >= rounds:
            break
        params[: len(shared)] = shared        # FedPer: base only; FedAvg: whole vector
        for _ in range(epochs):
            train_epoch(params, Xtr, ytr, lr, 32, pos_weight, rng)
        score = balanced_accuracy(params, Xte, yte)
        post_update(server, node, rnd, len(Xtr), score, params[: len(shared)])
        log.append((rnd, node, score))
        last = rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--nodes", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    log = []
    threads = [threading.Thread(target=run_node,
                                args=(i, args.server, args.rounds, args.epochs, args.lr, log))
               for i in range(args.nodes)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with urllib.request.urlopen(f"{args.server}/status") as r:
        st = json.loads(r.read())
    print(f"\nmode={st['mode']}  shared_params={st['n_shared']}  ({time.time() - t0:.1f}s)")
    print("round  nodes  balanced_acc(weighted)  bytes_in  bytes_out")
    for h in st["history"]:
        print(f"{h['round']:5d}  {h['n_updates']:5d}  {h['wavg_score']:22.3f}  "
              f"{h['bytes_in']:8d}  {h['bytes_out']:9d}")


if __name__ == "__main__":
    main()
