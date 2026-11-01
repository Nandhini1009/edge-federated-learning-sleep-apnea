#!/usr/bin/env python3
"""
FedApnea aggregator. Runs on your laptop now, on the Raspberry Pi later.
Dependencies: Python 3 + numpy only (no Flask/FastAPI to install).

    python fl_server.py --mode fedper --min-nodes 3

Endpoints (see PROTOCOL.md for the binary formats):
    GET  /round             -> {"round": N}   (cheap poll: has the round advanced?)
    GET  /model?node=ID     -> current global shared weights (binary)
    POST /update?node=ID    -> upload locally trained shared weights (binary)
    GET  /status            -> JSON: round, mode, per-round history (accuracy, bytes)
    POST /reset             -> restart training from round 0 (new run)

A round closes when --min-nodes updates have arrived, or --timeout seconds after
the first update of the round (whichever comes first). The new global vector is
the sample-count-weighted average of the received vectors.
"""

import argparse
import json
import sqlite3
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import spec

MODEL_HDR = struct.Struct("<4sIH")      # magic "FLM1", round u32, n_params u16
UPDATE_HDR = struct.Struct("<4sIIfH")   # magic "FLU1", round u32, n_samples u32, score f32, n_params u16


class State:
    def __init__(self, n_shared, mode, min_nodes, timeout, seed, db_path):
        self.lock = threading.Lock()
        self.n_shared = n_shared
        self.mode = mode
        self.min_nodes = min_nodes
        self.timeout = timeout
        self.seed = seed
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS rounds (run_id TEXT, mode TEXT, round INTEGER, "
            "n_updates INTEGER, wavg_score REAL, mean_score REAL, bytes_in INTEGER, "
            "bytes_out INTEGER, ts REAL)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS updates (run_id TEXT, round INTEGER, node TEXT, "
            "n_samples INTEGER, score REAL, bytes INTEGER, ts REAL)")
        self.reset()

    def reset(self):
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        full = spec.init_params(np.random.default_rng(self.seed))
        self.weights = full[: self.n_shared].copy()
        self.round = 0
        self.updates = {}            # node -> (n_samples, score, vector)
        self.first_update_ts = None
        self.bytes_in = 0
        self.bytes_out = 0
        self.history = []

    def aggregate(self):
        """Caller must hold self.lock."""
        nodes = list(self.updates)
        ns = np.array([self.updates[n][0] for n in nodes], dtype=np.float64)
        scores = np.array([self.updates[n][1] for n in nodes], dtype=np.float64)
        W = np.stack([self.updates[n][2] for n in nodes]).astype(np.float64)
        self.weights = (W * (ns / ns.sum())[:, None]).sum(axis=0).astype(np.float32)
        rec = dict(
            round=self.round,
            n_updates=len(nodes),
            wavg_score=float((scores * ns).sum() / ns.sum()),
            mean_score=float(scores.mean()),
            bytes_in=self.bytes_in,
            bytes_out=self.bytes_out,
            ts=time.time(),
        )
        self.history.append(rec)
        self.db.execute(
            "INSERT INTO rounds VALUES (?,?,?,?,?,?,?,?,?)",
            (self.run_id, self.mode, rec["round"], rec["n_updates"], rec["wavg_score"],
             rec["mean_score"], rec["bytes_in"], rec["bytes_out"], rec["ts"]))
        self.db.commit()
        print(f"[round {self.round:3d}] nodes={len(nodes)} "
              f"score(wavg)={rec['wavg_score']:.3f} in={rec['bytes_in']}B out={rec['bytes_out']}B")
        self.round += 1
        self.updates = {}
        self.first_update_ts = None
        self.bytes_in = 0
        self.bytes_out = 0


STATE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "FedApnea/0.1"

    def log_message(self, *args):
        pass  # keep the console readable

    def _send(self, code, body=b"", ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/model":
            with STATE.lock:
                body = MODEL_HDR.pack(b"FLM1", STATE.round, STATE.n_shared) \
                    + STATE.weights.astype("<f4").tobytes()
                STATE.bytes_out += len(body)
            self._send(200, body)
        elif u.path == "/round":
            # tiny endpoint for cheap polling (the ESP32 should poll this, not /model)
            with STATE.lock:
                self._json(200, dict(round=STATE.round))
        elif u.path == "/status":
            with STATE.lock:
                self._json(200, dict(
                    mode=STATE.mode, run_id=STATE.run_id, round=STATE.round,
                    n_shared=STATE.n_shared, min_nodes=STATE.min_nodes,
                    received=list(STATE.updates), history=STATE.history))
        elif u.path == "/":
            self._send(200, b"FedApnea aggregator OK. See /status\n", "text/plain")
        else:
            self._send(404, b"not found\n", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        node = q.get("node", ["?"])[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if u.path == "/reset":
            with STATE.lock:
                STATE.reset()
            self._json(200, dict(status="reset", run_id=STATE.run_id))
            return

        if u.path != "/update":
            self._send(404, b"not found\n", "text/plain")
            return

        if len(body) < UPDATE_HDR.size:
            self._json(400, dict(error="short_body"))
            return
        magic, rnd, n_samples, score, n_params = UPDATE_HDR.unpack_from(body, 0)
        payload = body[UPDATE_HDR.size:]
        if magic != b"FLU1" or n_params != STATE.n_shared or len(payload) != 4 * n_params:
            self._json(400, dict(error="bad_format", expected_params=STATE.n_shared))
            return
        vec = np.frombuffer(payload, dtype="<f4").astype(np.float32)
        if not np.all(np.isfinite(vec)) or n_samples == 0:
            self._json(400, dict(error="bad_values"))
            return

        with STATE.lock:
            if rnd != STATE.round:
                self._json(409, dict(error="stale_round", round=STATE.round))
                return
            if STATE.first_update_ts is None:
                STATE.first_update_ts = time.time()
            STATE.updates[node] = (n_samples, float(score), vec)
            STATE.bytes_in += len(body)
            STATE.db.execute(
                "INSERT INTO updates VALUES (?,?,?,?,?,?,?)",
                (STATE.run_id, rnd, node, n_samples, float(score), len(body), time.time()))
            got = len(STATE.updates)
            need = STATE.min_nodes
            if got >= need:
                STATE.aggregate()
                new_round = STATE.round
            else:
                new_round = STATE.round
        self._json(200, dict(status="ok", received=got, needed=need, round=new_round))


def watchdog():
    while True:
        time.sleep(0.5)
        with STATE.lock:
            if (STATE.updates and STATE.first_update_ts
                    and time.time() - STATE.first_update_ts > STATE.timeout):
                print(f"  timeout: aggregating {len(STATE.updates)} of {STATE.min_nodes} updates")
                STATE.aggregate()


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fedper", "fedavg"], default="fedper",
                    help="fedper: share base only; fedavg: share the whole model")
    ap.add_argument("--min-nodes", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds after the first update before closing a round anyway")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", default="fl.db")
    args = ap.parse_args()

    n_shared = spec.N_BASE if args.mode == "fedper" else spec.N_TOTAL
    STATE = State(n_shared, args.mode, args.min_nodes, args.timeout, args.seed, args.db)
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"FedApnea aggregator | mode={args.mode} shared_params={n_shared} "
          f"({n_shared * 4} B) min_nodes={args.min_nodes} port={args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
