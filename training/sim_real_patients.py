#!/usr/bin/env python3
"""
Runs real PhysioNet patients (from apnea_ecg_features.csv) as HTTP clients
against your LIVE fl_server.py - the same protocol sim_nodes.py already
proved works against your hardware - so they federate ALONGSIDE a real ESP32
node (esp32_fl_node_replay.ino) instead of only synthetic patients.

This is what turns the earlier solo b05-on-hardware run (which only showed
how weak that patient is in isolation, matching the standalone 0.439 Python
score) into a genuine test of pooling: b05's real on-device training combined
with real federated updates from the other patients, the same mechanism that
raised b05 from 0.273 (FedPer) to 0.494 (FedAvg) in fedavg_vs_fedper_real.py.

    python sim_real_patients.py apnea_ecg_features.csv --exclude b05 --rounds 20

Network functions (fetch_model/fetch_round/post_update) and the wire format
are copied UNCHANGED from sim_nodes.py, which has already run against your
real server successfully - only the data source changes (a real patient's
time-blocked split instead of make_patient()'s synthetic generator).

IMPORTANT - restart your server before running this:
    python fl_server.py --mode fedper --min-nodes <N>
where N = number of patients here (len(CLIENTS) printed below) + 1 for the
ESP32. Then reset the ESP32 and re-send its patient file so everyone starts
at round 0 together - otherwise the ESP32's earlier solo rounds and this
run's rounds won't line up.
"""
import argparse
import json
import struct
import threading
import time
import zlib
import urllib.error
import urllib.request

import numpy as np

import spec
from train_on_real_apnea import load_csv, zscore_fit, zscore_apply, train_epoch, evaluate
from fedavg_vs_fedper_real import pick_clients, pooled_scaling

MODEL_HDR = struct.Struct("<4sIH")
UPDATE_HDR = struct.Struct("<4sIIfH")


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


def run_node(record, Xtr, ytr, Xte, yte, server, rounds, epochs, lr, log):
    node = f"real-{record}"
    rng = np.random.default_rng(zlib.crc32(str(record).encode()))
    params = spec.init_params(rng)
    pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    last = -1
    while True:
        if last >= 0:
            while fetch_round(server) == last:
                time.sleep(0.05)
        rnd, shared = fetch_model(server, node)
        if rnd >= rounds:
            break
        params[: len(shared)] = shared
        idx = np.arange(len(Xtr))
        for _ in range(epochs):
            rng.shuffle(idx)
            train_epoch(params, Xtr[idx], ytr[idx], lr, 32, pos_weight, rng)
        score = evaluate(params, Xte, yte)["balanced_accuracy"]
        resp = post_update(server, node, rnd, len(Xtr), score, params[: len(shared)])
        print(f"  [{node}] round {rnd} posted, score={score:.3f}, "
              f"server now has {resp.get('received', '?')}/{resp.get('needed', '?')}", flush=True)
        log.append((rnd, node, score))
        last = rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="apnea_ecg_features.csv")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="records to skip (e.g. the one running on the real ESP32)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--local-scaling", action="store_true",
                    help="OLD behaviour: scale each patient with its own statistics (do not use if the "
                         "ESP32 or the model export uses the shared scaling)")
    args = ap.parse_args()

    records, y, X = load_csv(args.csv)
    all_clients, _ = pick_clients(records, y, X)          # the usable patients, same rule as everywhere else
    if args.local_scaling:
        print("scaling: LOCAL - each patient uses its own statistics (old behaviour)")
    else:
        mu, sd = pooled_scaling(all_clients)               # includes excluded records: they are nodes too
        for c in all_clients:
            c.apply_norm(mu, sd)
        print(f"scaling: SHARED - one mean/std pooled over the train rows of {len(all_clients)} patients "
              f"{[str(c.record) for c in all_clients]}; every node must use these same constants")
    clients = []
    for c in all_clients:
        rec = str(c.record)
        if rec in args.exclude:
            continue
        if args.local_scaling:
            mu_c, sd_c = zscore_fit(c.Xtr_raw)
            clients.append((rec, zscore_apply(c.Xtr_raw, mu_c, sd_c), c.ytr, zscore_apply(c.Xte_raw, mu_c, sd_c), c.yte))
        else:
            clients.append((rec, c.Xtr, c.ytr, c.Xte, c.yte))
    n_excluded = len([c for c in all_clients if str(c.record) in args.exclude])

    print(f"{len(clients)} real patients will run as nodes: {[c[0] for c in clients]}")
    print(f"excluded: {args.exclude}")
    print(f"=> start/restart your server with: python fl_server.py --mode fedavg "
          f"--min-nodes {len(clients) + n_excluded}")
    print("(the +1 per excluded record assumes it's running on a real ESP32 alongside this)\n")
    input("Press Enter once the server is running with that --min-nodes value "
          "and the ESP32 has been reset and re-sent its data, to start together...")

    log = []
    threads = [threading.Thread(target=run_node,
                                args=(rec, Xtr, ytr, Xte, yte, args.server, args.rounds,
                                      args.epochs, args.lr, log))
               for rec, Xtr, ytr, Xte, yte in clients]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with urllib.request.urlopen(f"{args.server}/status") as r:
        st = json.loads(r.read())
    print(f"\nmode={st['mode']}  shared_params={st['n_shared']}  ({time.time()-t0:.1f}s)")
    print("round  nodes  balanced_acc(weighted)  bytes_in  bytes_out")
    for h in st["history"]:
        print(f"{h['round']:5d}  {h['n_updates']:5d}  {h['wavg_score']:22.3f}  "
              f"{h['bytes_in']:8d}  {h['bytes_out']:9d}")

    print("\nper-patient score, last round each reported:")
    last_score = {}
    for rnd, node, score in log:
        last_score[node] = (rnd, score)
    for node, (rnd, score) in sorted(last_score.items()):
        print(f"  {node:>14s}: round {rnd}, balanced_acc={score:.3f}")


if __name__ == "__main__":
    main()
