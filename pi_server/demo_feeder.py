#!/usr/bin/env python3
"""
Pretend to be the ESP32: post real PhysioNet patient windows (the late-night test windows stored
in validation_set.h) to the dashboard server, one every few seconds, in time order.

    python demo_feeder.py --server http://<pi-ip>:8081 --patient a04 --interval 2

Uses: testing the Pi + dashboard before touching the sensor, and demonstrating the alert path
without electrodes. Each window carries the patient's TRUE label (shown on the dashboard log as
'truth' only in dashboard_log.csv). Windows are real; they come from a patient, not from you.
"""
import argparse
import json
import re
import time
import urllib.request
import zlib

import numpy as np


def block(text, name):
    m = re.search(rf"{name}[^=]*=\s*\{{(.*?)\}};", text, re.S)
    if not m:
        raise SystemExit(f"cannot find {name} in the header")
    return m.group(1)


def load_validation(path):
    t = open(path).read()
    names = re.findall(r'"([^"]+)"', block(t, "VAL_PATIENT_NAMES"))
    pid = np.array([int(v) for v in block(t, "VAL_PID").split(",") if v.strip()])
    y = np.array([int(v) for v in block(t, "VAL_Y").split(",") if v.strip()])
    pp = np.array([float(v.strip().rstrip("f")) for v in block(t, "VAL_P_PY").replace("\n", " ").split(",") if v.strip()])
    xb = re.search(r"VAL_X\[[^=]*=\s*\{(.*)\};", t, re.S).group(1)
    x = np.array([[float(v.strip().rstrip("f")) for v in row.split(",")] for row in re.findall(r"\{([^{}]*)\}", xb)])
    return names, pid, y, pp, x


def model_crc(path):
    t = open(path).read()

    def arr(name):
        return np.array([float(v.strip().rstrip("f")) for v in block(t, name).replace("\n", " ").split(",") if v.strip()],
                        dtype=np.float32)
    return "%08x" % (zlib.crc32(np.concatenate([arr("MODEL_W"), arr("FEAT_MU"), arr("FEAT_SD")]).astype("<f4").tobytes()) & 0xFFFFFFFF)


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--header", default="esp32_model_validate/validation_set.h")
    ap.add_argument("--model", default="esp32_model_validate/model_export.h")
    ap.add_argument("--patient", default="a04")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between windows (the real board sends every 10)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0 = all windows of that patient")
    ap.add_argument("--device", default="demo-feeder")
    ap.add_argument("--alert-demo", action="store_true",
                    help="find the longest stretch of REAL apnea windows that the model also flags, and replay it "
                         "(with a few normal windows before and after) so the dashboard raises a real ALERT")
    ap.add_argument("--threshold", type=float, default=0.7, help="alert threshold used by --alert-demo (match the server)")
    ap.add_argument("--no-signal", default="", help="window numbers to send as 'no usable signal', e.g. 10-13")
    a = ap.parse_args()

    names, pid, y, pp, x = load_validation(a.header)
    if a.alert_demo:
        sustain = 6
        good = (y == 1) & (pp >= a.threshold)
        best = (0, 0, -1)                       # (length, start index, patient)
        for k in range(len(names)):
            ii = np.where(pid == k)[0]
            run = 0
            for j, i in enumerate(ii):
                run = run + 1 if good[i] else 0
                if run > best[0]:
                    best = (run, ii[j - run + 1], k)
        if best[0] >= sustain:
            k = best[2]
            ii = np.where(pid == k)[0]
            s0 = int(np.where(ii == best[1])[0][0])
            lo, hi = max(0, s0 - 3), min(len(ii), s0 + min(best[0], 14) + 4)
            idx = ii[lo:hi]
            print(f"alert demo: patient {names[k]} has {best[0]} real apnea windows in a row that the model flags "
                  f"(probability >= {a.threshold}); replaying {len(idx)} windows around them")
        else:
            idx = np.where(good)[0][:14]
            print(f"alert demo: no {sustain}-in-a-row stretch exists at threshold {a.threshold}; replaying the "
                  f"{len(idx)} real apnea windows the model flags most strongly (not consecutive in time)")
            if len(idx) < sustain:
                raise SystemExit("not enough flagged windows for an alert; lower --threshold on both the feeder and the server")
        a.patient = "(alert demo)"
    else:
        if a.patient not in names:
            raise SystemExit(f"patient '{a.patient}' not in {names}")
        k = names.index(a.patient)
        idx = np.where(pid == k)[0][a.start:]
        if a.count:
            idx = idx[:a.count]
    crc = model_crc(a.model)
    gap = set()
    if a.no_signal:
        lo, _, hi = a.no_signal.partition("-")
        gap = set(range(int(lo), int(hi or lo) + 1))
    print(f"feeding {len(idx)} real windows of patient {a.patient} to {a.server} every {a.interval} s (model {crc})")

    ack = 0
    for n, i in enumerate(idx):
        if n in gap:
            rep = dict(device=a.device, t=n * 10, valid=0, why="lead-off or flat signal inside the window", ack=ack, crc=crc)
        else:
            rep = dict(device=a.device, t=n * 10, valid=1, f=[float(v) for v in x[i]], bp=float(pp[i]),
                       truth=int(y[i]), ack=ack, crc=crc, rssi=-50)
        r = post(a.server.rstrip("/") + "/ingest", rep)
        ack = r["level"]
        truth = "" if n in gap else f"truth={'apnea' if y[i] else 'normal'}"
        print(f"  window {n:3d}: Pi probability {r['p']:.3f} -> {r['state']:9s} (level {r['level']})  {truth}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
