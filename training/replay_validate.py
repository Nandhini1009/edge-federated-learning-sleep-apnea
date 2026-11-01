#!/usr/bin/env python3
"""
Replay REAL PhysioNet ECG through the ESP32's live pipeline and check it.

For each record it takes a stretch of real ECG from the LATE part of the night
(the model was trained on the early part of each patient's night), resamples it
to 250 Hz exactly like apnea_ecg_extract.py did, and then:

  1. Python works out what it should get: beats -> 60 s windows -> 8 features
     -> scaling -> model probability (apnea_ecg_extract.py's own functions and
     the same weights as model_export.h).
  2. The same samples are streamed to the board (esp32_live_infer.ino compiled
     with REPLAY_MODE 1). The board runs its beat detector, features and model
     and sends back one result per minute.
  3. The two are compared window by window: same windows valid? same beat
     counts? same features? same probability?
  4. The board's decision is also compared with the TRUE apnea/normal label of
     each annotated minute. This is a small sample, a sanity check only.

    pip install pyserial wfdb
    python replay_validate.py COM5 --records a05 b03 --minutes 10

Close the Arduino Serial Monitor first (one program per COM port).
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apnea_ecg_extract as ex
from train_on_real_apnea import forward

FS = 250
PREROLL_S = 10      # signal before the first scored minute (detector warm-up)
WIN_S = 60
Q = 1_000_000       # samples travel as integer micro-millivolts; BOTH sides use q / 1e6
CHUNK = 500         # the board ACKs every 500 samples


def parse_header(path):
    s = open(path).read()

    def arr(name):
        m = re.search(rf"static const float {name}\[\d+\] = \{{(.*?)\}};", s, re.S)
        return np.array([float(t.strip().rstrip("f")) for t in m.group(1).replace("\n", " ").split(",")
                         if t.strip()], dtype=np.float32)
    return arr("MODEL_W"), arr("FEAT_MU"), arr("FEAT_SD")


# ---------------------------------------------------------------- data
def load_real(name):
    import wfdb
    rec = wfdb.rdrecord(name, pn_dir="apnea-ecg")
    ann = wfdb.rdann(name, "apn", pn_dir="apnea-ecg")
    return dict(name=name, sig=rec.p_signal[:, 0].astype(np.float64), fs0=float(rec.fs),
                starts=[s / rec.fs for s in ann.sample], labels=list(ann.symbol))


def load_synthetic(name, seed, n_min=30):
    rng = np.random.default_rng(seed)
    fs0 = 100.0
    labels = ["A" if (m % 3 == 1) else "N" for m in range(n_min)]
    n = int(n_min * 60 * fs0)
    t = np.arange(n) / fs0
    sig = np.zeros(n)
    tb = 0.4
    while tb < n_min * 60 - 0.4:
        a = labels[int(tb // 60)] == "A"
        i0, i1 = max(0, int((tb - 0.5) * fs0)), min(n, int((tb + 0.6) * fs0))
        tt = t[i0:i1]
        sig[i0:i1] += (0.15 * np.exp(-0.5 * ((tt - (tb - 0.16)) / 0.025) ** 2)
                       + np.exp(-0.5 * ((tt - tb) / 0.012) ** 2)
                       + 0.25 * np.exp(-0.5 * ((tt - (tb + 0.28)) / 0.05) ** 2))
        tb += 60.0 / ((55 + rng.uniform(-12, 12)) if a else (72 + rng.uniform(-5, 5)))
    sig += rng.normal(0, 0.02, n)
    return dict(name=name, sig=sig, fs0=fs0, starts=[m * 60.0 for m in range(n_min)], labels=labels)


def make_snippet(rec, minutes, start_frac):
    """Real ECG from annotated minute m0 onward, with PREROLL_S of signal before it."""
    sig250 = ex.resample_to_250(rec["sig"], rec["fs0"])
    n_ann = len(rec["labels"])
    m0 = int(start_frac * n_ann)
    k = min(minutes, n_ann - m0 - 1)
    i0 = int(round((rec["starts"][m0] - PREROLL_S) * FS))
    i1 = i0 + (PREROLL_S + k * WIN_S) * FS + 1      # +1: a beat is decided one sample after it happens
    if i0 < 0 or i1 > len(sig250) or k < 2:
        return None
    seg = sig250[i0:i1]
    if not np.isfinite(seg).all():
        print(f"  {rec['name']}: NaN in the chosen stretch - skipped")
        return None
    q = np.rint(seg * Q).astype(np.int64)
    return dict(name=rec["name"], q=q, labels=rec["labels"][m0:m0 + k], m0=m0, k=k)


# ---------------------------------------------------------------- what Python says it should be
def py_reference(q, k, W, MU, SD):
    x = q / Q                                         # identical doubles on both sides
    long_bad = ex.find_long_bad_stretches(x, FS)
    _, rr_pairs = ex.detect_r_peaks_segmented(x, FS)
    out = []
    for w in range(k):
        t1 = float(PREROLL_S + WIN_S * (w + 1))
        t0 = t1 - WIN_S
        i0, i1 = int(t0 * FS), int(t1 * FS)
        if long_bad[i0:i1].any():
            out.append(None); continue
        rr = np.array([r for t, r in rr_pairs if t0 <= t < t1])
        rr = rr[(rr >= 0.3) & (rr <= 2.0)]
        if len(rr) < 20 or rr.sum() / WIN_S < 0.70:
            out.append(None); continue
        f = np.array(ex.hrv_features(rr), dtype=np.float32)
        xn = ((f - MU) / SD).astype(np.float32)
        out.append(dict(n=len(rr), cov=float(rr.sum() / WIN_S), f=f,
                        p=float(forward(W, xn[None, :]).reshape(-1)[0])))
    return out


# ---------------------------------------------------------------- the board (real USB or a native build for testing)
class SerialBoard:
    def __init__(self, port):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=2)
        self.ser.reset_input_buffer()
        self.ser.setDTR(False); time.sleep(0.2); self.ser.setDTR(True)   # clean reset
        self.wait("READY-REPLAY", 25)

    def send(self, s):
        self.ser.write(s.encode())

    def readline(self):
        return self.ser.readline().decode(errors="replace").strip()

    def close(self):
        self.ser.close()

    def wait(self, token, timeout, sink=None):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.readline()
            if sink is not None and line.startswith("RESULT"):
                sink.append(line)
            if token in line:
                return
        raise RuntimeError(f"timed out waiting for '{token}' from the board - is the sketch built with "
                           f"REPLAY_MODE 1, and is the Serial Monitor closed?")


class NativeBoard(SerialBoard):
    def __init__(self, binary):
        self.p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.wait("READY-REPLAY", 20)

    def send(self, s):
        self.p.stdin.write(s); self.p.stdin.flush()

    def readline(self):
        return self.p.stdout.readline().strip()

    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=10)


def run_on_board(board, q, tamper=None):
    lines = [str(int(v)) for v in q]
    if tamper:                                        # self-test only: corrupt what the board receives
        for idx, val in tamper:
            lines[idx] = str(val)
    sink = []
    board.send("R\n"); board.wait("RESET_OK", 10, sink)
    t0 = time.time()
    for s in range(0, len(lines), CHUNK):
        chunk = lines[s:s + CHUNK]
        board.send("\n".join(chunk) + "\n")
        if len(chunk) == CHUNK:
            board.wait("ACK", 60, sink)
        if (s // CHUNK) % 60 == 0:
            print(f"    streamed {s}/{len(lines)} samples ({time.time() - t0:.0f} s)", flush=True)
    board.send("E\n"); board.wait("REPLAY_DONE", 30, sink)
    res = {}
    for line in sink:
        t = line.split()
        idx, valid, n, cov = int(t[1]), t[2] == "1", int(t[3]), float(t[4])
        res[idx] = dict(valid=valid, n=n, cov=cov,
                        f=np.array(t[5:13], dtype=np.float64) if valid else None,
                        p=float(t[13]) if valid else None)
    return res


# ---------------------------------------------------------------- compare
def compare(name, ref, brd, labels):
    ok = True
    maxf = maxp = maxcov = 0.0
    flips = nmis = vmis = missing = 0
    tp = tn = fp = fn = 0
    print(f"\n  {name}: minute | true label | python p | board p | board says")
    for w, r in enumerate(ref):
        b = brd.get(w)
        if b is None:
            print(f"    {w:3d}  (no result from the board)"); missing += 1; continue
        if (r is None) != (not b["valid"]):
            vmis += 1
            print(f"    {w:3d}  {labels[w]}   validity differs: python {'valid' if r else 'invalid'}, "
                  f"board {'valid' if b['valid'] else 'invalid'}")
            continue
        if r is None:
            print(f"    {w:3d}  {labels[w]}   not scored (too few or too sparse beats) - same on both")
            continue
        nmis += int(r["n"] != b["n"])
        maxcov = max(maxcov, abs(r["cov"] - b["cov"]))
        maxf = max(maxf, float(np.max(np.abs(r["f"].astype(np.float64) - b["f"]) / (np.abs(b["f"]) + 1e-9))))
        maxp = max(maxp, abs(r["p"] - b["p"]))
        flip = (r["p"] >= 0.5) != (b["p"] >= 0.5)
        flips += int(flip and abs(r["p"] - 0.5) > 1e-4)
        dec = "APNEA" if b["p"] >= 0.5 else "normal"
        lab = labels[w]
        if lab in ("A", "N"):
            truth = lab == "A"; pred = b["p"] >= 0.5
            tp += truth and pred; tn += (not truth) and (not pred)
            fp += (not truth) and pred; fn += truth and (not pred)
        print(f"    {w:3d}  {lab}   {r['p']:.4f}   {b['p']:.4f}   {dec}")
    good = (missing == 0 and vmis == 0 and nmis == 0 and flips == 0 and maxp < 1e-4 and maxf < 1e-5 and maxcov < 1e-9)
    print(f"  {name}: windows valid on both sides, beat counts equal, largest feature difference "
          f"{maxf:.1e} (relative), probability {maxp:.1e}, decision flips {flips}, validity mismatches {vmis}, missing {missing}"
          f" -> {'MATCH' if good else 'MISMATCH'}")
    return good, (tp, tn, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None)
    ap.add_argument("--records", nargs="+", default=["a05", "b03"])
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--start-frac", type=float, default=0.80,
                    help="how far into the night to start (0.8 = late, where the model's test windows are)")
    ap.add_argument("--header", default=None, help="model_export.h (default: the one in esp32_live_infer or esp32_model_validate)")
    ap.add_argument("--synthetic", action="store_true", help="self-test with fake records, no internet")
    ap.add_argument("--native", default=None, help="self-test against a compiled copy of the sketch")
    ap.add_argument("--tamper", action="store_true", help="self-test: corrupt samples sent to the board (must FAIL)")
    a = ap.parse_args()

    header = a.header
    for cand in ([header] if header else ["esp32_live_infer/model_export.h", "esp32_model_validate/model_export.h"]):
        if cand and os.path.exists(cand):
            header = cand; break
    if not header or not os.path.exists(header):
        sys.exit("cannot find model_export.h - pass --header <path>")
    W, MU, SD = parse_header(header)
    print(f"model: {header}")

    snippets = []
    for i, name in enumerate(a.records):
        print(f"loading {name} ...")
        try:
            rec = load_synthetic(name, 100 + i) if a.synthetic else load_real(name)
        except Exception as e:
            print(f"  skipped {name}: {e}"); continue
        sn = make_snippet(rec, a.minutes, a.start_frac)
        if sn: snippets.append(sn)
    if not snippets:
        sys.exit("nothing to replay")

    board = NativeBoard(a.native) if a.native else SerialBoard(a.port)
    allok = True
    T = [0, 0, 0, 0]
    try:
        for sn in snippets:
            print(f"\n{sn['name']}: replaying minutes {sn['m0']}..{sn['m0'] + sn['k'] - 1} "
                  f"({len(sn['q'])} samples)")
            ref = py_reference(sn["q"], sn["k"], W, MU, SD)
            tamper = [(PREROLL_S * FS + 2000, int(sn["q"][PREROLL_S * FS + 2000]) + 3_000_000),
                      (PREROLL_S * FS + 9000, int(sn["q"][PREROLL_S * FS + 9000]) - 4_000_000)] if a.tamper else None
            brd = run_on_board(board, sn["q"], tamper)
            good, c = compare(sn["name"], ref, brd, sn["labels"])
            allok &= good
            T = [T[i] + c[i] for i in range(4)]
    finally:
        board.close()

    tp, tn, fp, fn = T
    n = sum(T)
    print("\n================ SUMMARY ================")
    print(f"board vs Python on real ECG: {'ALL WINDOWS MATCH' if allok else 'MISMATCH - see above'}")
    if n:
        print(f"board's decisions vs the true labels over {n} scored minutes: "
              f"correct {tp + tn}, wrong {fp + fn}  (apnea caught {tp}/{tp + fn}, normal kept {tn}/{tn + fp})")
        print("That is a handful of minutes from two patients, enough to show the whole path works end to end "
              "on real recordings, not enough to measure accuracy (the model's measured accuracy is ~0.65-0.68).")
    print("RESULT:", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
