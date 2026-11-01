#!/usr/bin/env python3
"""
FedApnea: build a REAL labeled dataset from the PhysioNet Apnea-ECG database.

Run this on a machine with internet access (it downloads the records from PhysioNet).

    pip install wfdb numpy
    python apnea_ecg_extract.py

What it does, per record:
    1. downloads the ECG signal (100 Hz) and the per-minute apnea/normal
       annotations from PhysioNet
    2. resamples the signal to 250 Hz (the rate every ESP32 sketch in this
       project is tuned for - INTEG_WIN_S, REFRACTORY_S etc. all assume it)
    3. runs the EXACT streaming R-peak detector from hrv_parity_ref.py /
       esp32_hrv_streaming_test.ino (same constants, same algorithm) over
       the whole resampled signal
    4. for each 1-minute annotation window, gathers the RR intervals whose
       peak falls inside that minute and computes the same 8 HRV features
       used everywhere else in this project
    5. writes one row per (record, minute) to apnea_ecg_features.csv:
       record,minute,label,mean_rr_s,sdnn_s,rmssd_s,pnn50,cv_rr,rr_range_s,
       median_rr_s,mean_abs_diff_s
       label: 1 = apnea ('A'), 0 = normal ('N')

Windowing convention (state this in your methodology): a minute's label is
paired with RR intervals whose peak time falls in [minute_start, minute_start
+ 60s), where minute_start comes directly from the annotation's sample index.
This is the standard convention used in Apnea-ECG literature; if you use a
different alignment elsewhere, keep it consistent everywhere you compute
features.

Records: by default the original 15 (a01-a05, b01-b05, c01-c05). Pick others on the command line:

    python apnea_ecg_extract.py --append --records a06-a20 c06-c10

--append keeps every row already in apnea_ecg_features.csv byte for byte (a backup is made), adds only
the records that are not in it yet, and saves after every record, so a long run can be stopped and
resumed by running the same command again. Without --append or --overwrite the script refuses to touch
an existing CSV.
"""

import csv
import sys

import numpy as np

# ---------------------------------------------------------------- detector
# Identical constants and algorithm to hrv_parity_ref.py's detect_r_peaks_causal
# and esp32_hrv_streaming_test.ino - do not change these without re-verifying
# everywhere else in the project.
FS = 250
INTEG_WIN_S = 0.04
REFRACTORY_S = 0.25
RR_MIN_S = 0.3
RR_MAX_S = 2.0


def interpolate_gaps(sig, max_gap_s, fs):
    """
    Linearly interpolate over non-finite (NaN/inf) stretches shorter than
    max_gap_s. Longer stretches are left as NaN - the caller must exclude any
    annotation window that overlaps a remaining NaN, since that much missing
    signal cannot be honestly reconstructed.

    THIS FIXES A REAL BUG: the streaming detector's running accumulator does
    `acc += new; acc -= old`, so a single non-finite sample anywhere in a
    multi-hour recording poisons acc with NaN PERMANENTLY (NaN - x = NaN
    forever after), silently destroying every beat detection for the rest of
    the recording. Confirmed experimentally: one injected NaN sample cut
    detected beats in the following 90 s from 112 to 0. This is almost
    certainly why a03 and c01 gave near-zero usable minutes. Interpolating
    (or otherwise never handing the detector a non-finite sample) is
    mandatory, not optional, for any record with real-world dropouts.
    """
    sig = np.asarray(sig, dtype=np.float64).copy()
    bad = ~np.isfinite(sig)
    if not bad.any():
        return sig, np.zeros_like(bad)
    max_gap_samples = int(max_gap_s * fs)

    # find contiguous bad runs
    idx = np.where(bad)[0]
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = i
        prev = i
    runs.append((start, prev))

    unresolved = np.zeros_like(bad)
    good_idx = np.where(np.isfinite(sig))[0]
    for s, e in runs:
        length = e - s + 1
        if length > max_gap_samples or len(good_idx) < 2:
            unresolved[s:e + 1] = True
            continue
        # linear interpolation using the nearest finite samples on each side
        left = good_idx[good_idx < s]
        right = good_idx[good_idx > e]
        if len(left) == 0 or len(right) == 0:
            unresolved[s:e + 1] = True
            continue
        l, r = left[-1], right[0]
        sig[s:e + 1] = np.interp(np.arange(s, e + 1), [l, r], [sig[l], sig[r]])

    if unresolved.any():
        # never let the detector see a NaN, even in an unresolved stretch -
        # flat-fill so acc can't be poisoned; the corresponding time range is
        # returned so the caller can exclude any window that overlaps it
        sig[unresolved] = 0.0
    return sig, unresolved


def find_long_bad_stretches(sig, fs, max_gap_s=2.0, min_stuck_s=2.0):
    """
    Finds stretches that are either non-finite OR an exactly-repeated value
    (a flat/stuck signal - the signature of a sensor/electrode dropout), and
    returns a boolean mask marking only the ones LONGER than max_gap_s.

    WHY THIS EXISTS: several real PhysioNet records (b04, c01, c02, c04) have
    100-300 SECOND stretches of an exactly repeated value - not NaN, so the
    earlier NaN-based gap handling never saw them. Flat-filling and
    continuing the same detector pass through a multi-minute dropout is
    unsafe for a different reason than the amplitude-outlier bug: the
    transition BACK to real signal when the dropout ends looks like a huge
    one-sample jump to the derivative-based detector, which can re-trigger
    the same threshold-lockout problem the percentile/cap fix addresses for
    isolated spikes, except now recurring every time a multi-minute dropout
    ends across a multi-hour recording. The robust fix is to never run the
    detector continuously across such a stretch at all - see extract_record,
    which uses this mask to split the signal into independent segments, each
    with its own fresh warmup, rather than patching the threshold maths
    further.
    """
    n = len(sig)
    nonfinite = ~np.isfinite(sig)

    finite_sig = np.where(np.isfinite(sig), sig, np.nan)
    same_as_prev = np.zeros(n, dtype=bool)
    same_as_prev[1:] = (finite_sig[1:] == finite_sig[:-1]) & np.isfinite(finite_sig[1:])

    bad = nonfinite.copy()
    idx = np.where(same_as_prev)[0]
    if len(idx):
        start = idx[0]; prev = idx[0]
        runs = []
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            runs.append((start - 1, prev))   # include the sample the run matches against
            start = i; prev = i
        runs.append((start - 1, prev))
        min_stuck_samples = int(min_stuck_s * fs)
        for s, e in runs:
            if (e - s + 1) >= min_stuck_samples:
                bad[s:e + 1] = True

    long_bad = np.zeros(n, dtype=bool)
    idx = np.where(bad)[0]
    if len(idx):
        start = idx[0]; prev = idx[0]
        runs = []
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            runs.append((start, prev))
            start = i; prev = i
        runs.append((start, prev))
        max_gap_samples = int(max_gap_s * fs)
        for s, e in runs:
            if (e - s + 1) > max_gap_samples:
                long_bad[s:e + 1] = True
    return long_bad


def detect_r_peaks_segmented(sig, fs=FS, max_gap_s=2.0, min_stuck_s=2.0):
    """
    Splits sig at any long NaN or stuck-value stretch (see
    find_long_bad_stretches) and runs detect_r_peaks_causal INDEPENDENTLY on
    each clean segment, each getting its own fresh 2 s warmup - the same
    thing a real device does after a lead-off/reattachment event. Returns
    (peak_times_s, rr_pairs) where rr_pairs is a list of (time_of_later_beat,
    rr_seconds) computed ONLY within a segment, never across a gap boundary
    (diffing across a boundary would fabricate one fake multi-minute "RR"
    interval per gap).
    """
    bad = find_long_bad_stretches(sig, fs, max_gap_s, min_stuck_s)
    n = len(sig)
    good = ~bad
    segments = []
    if good.any():
        idx = np.where(good)[0]
        start = idx[0]; prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            segments.append((start, prev))
            start = i; prev = i
        segments.append((start, prev))

    min_len = int(2.5 * fs)   # shorter than this can't even complete a warmup
    all_peak_times = []
    rr_pairs = []
    for s, e in segments:
        if e - s + 1 < min_len:
            continue
        seg = sig[s:e + 1]
        seg_clean, _ = interpolate_gaps(seg, max_gap_s, fs)   # mop up any short NaN blips within the segment
        peaks = detect_r_peaks_causal(seg_clean, fs)
        times = (peaks + s) / fs
        all_peak_times.append(times)
        if len(times) > 1:
            rr = np.diff(times)
            for t, r in zip(times[1:], rr):
                rr_pairs.append((t, r))

    peak_times = np.concatenate(all_peak_times) if all_peak_times else np.array([])
    return peak_times, rr_pairs


SPKI_MAX_JUMP = 5.0   # see the fix note below: caps how much one outlier can inflate spki


def detect_r_peaks_causal(sig, fs=FS):
    """
    Causal detector, with a robustness fix on top of the originally verified
    algorithm (see hrv_parity_ref.py for that baseline - it passed 76/76 on a
    clean synthetic recording and is unchanged in spirit here).

    THE BUG THIS FIXES: the adaptive threshold used raw max()/EMA-of-accepted-
    peaks statistics, which have no defense against a single extreme-amplitude
    artifact (a motion/electrode-pop spike, common in real ambulatory ECG).
    One such spike during the 2 s warmup window set spki (and therefore the
    threshold) using an uncapped max(), and since spki only ever updates via
    NEW accepted peaks, an artifact that inflates it once can lock the
    detector out for a long time (in a repro test: recurring 10x-amplitude
    spikes every 20 s cut detection from 372/375 true beats to 14/375 -
    matching the near-total detection loss seen on several real PhysioNet
    records that turned out to have rare but very large-amplitude outliers).

    Two changes, both standard robust-statistics fixes for exactly this
    failure mode (conceptually similar to the "search-back" mechanism in the
    original Pan-Tompkins paper, simplified for this streaming context):
      1. warmup uses the 95th percentile of the warmup window's integrated
         signal, not its literal maximum, so one warmup-period outlier can't
         set a runaway-high initial threshold.
      2. any single peak candidate's contribution to spki is capped at
         SPKI_MAX_JUMP x the current spki, so no later outlier can multiply
         the threshold out of reach either.
    Re-verified against the ORIGINAL 76/76-beat golden test after this change
    (see verify_detector_robustness_fix.py) to confirm no regression on clean
    signal - both changes are no-ops when there are no outliers.
    """
    n = len(sig)
    win = max(1, int(INTEG_WIN_S * fs))
    refractory = int(REFRACTORY_S * fs)
    warmup_target = 2 * fs

    sq_ring = np.zeros(win, dtype=np.float64)
    ring_pos = 0
    acc = 0.0
    prev_sample = 0.0

    warmup_vals = np.empty(warmup_target, dtype=np.float64)
    warmup_sum = 0.0
    warmup_count = 0
    warmed_up = False
    spki = npki = threshold = 0.0

    prev_integ = None
    prev_is_rise = False
    last_peak = -refractory
    peaks = []

    for i in range(n):
        diff = 0.0 if i == 0 else (sig[i] - prev_sample)
        prev_sample = sig[i]
        sq = diff * diff
        acc += sq
        if i >= win:
            acc -= sq_ring[ring_pos]
        sq_ring[ring_pos] = sq
        ring_pos = (ring_pos + 1) % win
        integ_i = acc / win

        if not warmed_up:
            if warmup_count < warmup_target:
                warmup_vals[warmup_count] = integ_i
            warmup_sum += integ_i
            warmup_count += 1
            if warmup_count >= warmup_target:
                robust_max = np.percentile(warmup_vals, 95)   # fix #1: percentile, not max()
                spki = robust_max * 0.5
                npki = warmup_sum / warmup_count
                threshold = npki + 0.25 * (spki - npki)
                warmed_up = True
        elif prev_integ is not None and i >= 2:
            if prev_is_rise and prev_integ >= integ_i:
                peak_idx = i - 1
                peak_val = prev_integ
                if peak_val > threshold and peak_idx - last_peak > refractory:
                    last_peak = peak_idx
                    capped_val = min(peak_val, SPKI_MAX_JUMP * spki) if spki > 0 else peak_val  # fix #2
                    spki = 0.125 * capped_val + 0.875 * spki
                    peaks.append(peak_idx)
                else:
                    npki = 0.125 * peak_val + 0.875 * npki
                threshold = npki + 0.25 * (spki - npki)

        if prev_integ is not None:
            prev_is_rise = (integ_i > prev_integ) if warmed_up else False
        prev_integ = integ_i

    return np.array(peaks, dtype=np.int64)


def hrv_features(rr):
    """Identical to hrv_features() in hrv_parity_ref.py. rr in seconds."""
    rr = np.asarray(rr, dtype=np.float64)
    diffs = np.diff(rr)
    mean_rr = rr.mean()
    sdnn = rr.std(ddof=0)
    rmssd = np.sqrt(np.mean(diffs ** 2)) if len(diffs) else 0.0
    pnn50 = np.mean(np.abs(diffs) > 0.05) if len(diffs) else 0.0
    cv_rr = sdnn / mean_rr if mean_rr > 0 else 0.0
    rr_range = rr.max() - rr.min()
    median_rr = np.median(rr)
    mean_abs_diff = np.mean(np.abs(diffs)) if len(diffs) else 0.0
    return [mean_rr, sdnn, rmssd, pnn50, cv_rr, rr_range, median_rr, mean_abs_diff]


FEATURE_NAMES = ["mean_rr_s", "sdnn_s", "rmssd_s", "pnn50", "cv_rr",
                  "rr_range_s", "median_rr_s", "mean_abs_diff_s"]


def resample_to_250(sig, orig_fs):
    """Simple linear-interpolation resample. sig: 1-D array at orig_fs Hz."""
    n_out = int(round(len(sig) * FS / orig_fs))
    t_in = np.arange(len(sig)) / orig_fs
    t_out = np.arange(n_out) / FS
    return np.interp(t_out, t_in, sig)


def extract_record(sig_250_raw, ann_sample_orig, ann_label, orig_fs, min_rr_per_window=20,
                    max_gap_s=2.0, min_stuck_s=2.0, verbose_name=None):
    """
    sig_250_raw: ECG at 250 Hz (already resampled) - MAY contain NaN/inf
        and/or long flat/stuck-value dropouts; this function handles both
        (see find_long_bad_stretches / detect_r_peaks_segmented), so callers
        no longer need to pre-clean it.
    ann_sample_orig: annotation sample indices IN THE ORIGINAL sampling rate.
    ann_label: 'A' or 'N' per annotation (one per minute).
    Returns a list of (label01, [8 features]).
    """
    long_bad = find_long_bad_stretches(sig_250_raw, FS, max_gap_s, min_stuck_s)
    if verbose_name and long_bad.any():
        pct = 100 * long_bad.sum() / len(long_bad)
        print(f"    {verbose_name}: {pct:.1f}% of the record is a long NaN or stuck-value "
              f"dropout (>{max_gap_s}s) - the signal is segmented around it, windows "
              f"overlapping it are excluded")

    peak_times, rr_pairs = detect_r_peaks_segmented(sig_250_raw, FS, max_gap_s, min_stuck_s)

    rows = []
    for samp, lab in zip(ann_sample_orig, ann_label):
        if lab not in ("A", "N"):
            continue
        t0 = samp / orig_fs
        t1 = t0 + 60.0
        i0, i1 = int(t0 * FS), int(t1 * FS)
        if long_bad[max(i0, 0):min(i1, len(long_bad))].any():
            continue   # this minute overlaps signal we could not honestly reconstruct
        rr = np.array([r for t, r in rr_pairs if t0 <= t < t1])
        rr = rr[(rr >= RR_MIN_S) & (rr <= RR_MAX_S)]
        if len(rr) < min_rr_per_window:
            continue
        feats = hrv_features(rr)
        rows.append((1 if lab == "A" else 0, feats))
    return rows


# ---------------------------------------------------------------- real download + run
# Widened and rebalanced across all three PhysioNet apnea-ecg groups:
#   a-group = apnea-heavy patients, b-group = borderline, c-group = mostly normal.
# The earlier 6-record pull was 4 a's + 1 b + 1 c, which skewed training data
# to ~90% apnea while a held-out b/c patient looks ~5% apnea - a real domain-
# shift problem, not a bug (see the b01 test result: high sensitivity, poor
# precision). This 15-record pull (5 per group) gives both a more realistic
# overall prevalence and enough records per group for leave-one-record-out
# cross-validation to mean something.
RECORDS = ["a01", "a02", "a03", "a04", "a05",
           "b01", "b02", "b03", "b04", "b05",
           "c01", "c02", "c03", "c04", "c05"]


def install_download_timeouts():
    """The download library (wfdb, via `requests`) sets NO timeout, so a stalled connection waits forever and the whole job
    freezes. Give every request a connect timeout and a read timeout; a stall then raises 'timed out', which the retry
    logic in run_real already handles. Only requests that set no timeout of their own are touched."""
    import os
    try:
        import requests
    except ImportError:
        return
    if getattr(requests.adapters.HTTPAdapter.send, "_apnea_patched", False):
        return
    connect = float(os.environ.get("APNEA_CONNECT_TIMEOUT", "20"))
    read = float(os.environ.get("APNEA_READ_TIMEOUT", "90"))
    original = requests.adapters.HTTPAdapter.send

    def send(self, request, **kw):
        if kw.get("timeout") is None:
            kw["timeout"] = (connect, read)
        return original(self, request, **kw)

    send._apnea_patched = True
    requests.adapters.HTTPAdapter.send = send


def expand_records(tokens):
    """'a06-a20' -> a06 ... a20 (same letter, zero-padded numbers); plain names pass through."""
    import re
    out = []
    for t in tokens:
        t = t.strip().lower()
        m = re.fullmatch(r"([a-z]+)(\d+)-(?:[a-z]+)?(\d+)", t)
        if m:
            letter, lo, hi = m.group(1), m.group(2), m.group(3)
            if int(hi) < int(lo):
                raise SystemExit(f"cannot read the range '{t}': the end is smaller than the start")
            out += [f"{letter}{n:0{len(lo)}d}" for n in range(int(lo), int(hi) + 1)]
        elif "-" in t:
            raise SystemExit(f"cannot read '{t}' as a range; write it like a06-a20")
        else:
            out.append(t)
    return out


def run_real(records, out_path, mode, redo):
    """mode: 'new' (refuse to touch an existing file), 'overwrite', or 'append' (keep existing rows, add new records)."""
    import os
    import shutil
    import wfdb
    install_download_timeouts()
    header = ["record", "label"] + FEATURE_NAMES
    rows_out = []                     # rows already in the file, kept EXACTLY as written (never re-parsed)
    done = set()
    if os.path.exists(out_path):
        if mode == "new":
            raise SystemExit(f"{out_path} already exists. Add records to it with --append, or rebuild it with --overwrite.")
        if mode == "append":
            with open(out_path, newline="") as f:
                rd = csv.reader(f)
                next(rd)
                rows_out = [r for r in rd]
            done = {r[0] for r in rows_out}
            bak = out_path + ".bak"
            n = 2
            while os.path.exists(bak):                                   # never overwrite an earlier backup
                bak = f"{out_path}.bak{n}"; n += 1
            shutil.copy(out_path, bak)
            print(f"keeping the {len(rows_out)} rows already in {out_path} ({len(done)} records); backup: {bak}")
    if redo:
        rows_out = [r for r in rows_out if r[0] not in set(records)]
        done -= set(records)

    def save():
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_out)

    todo = [r for r in records if r not in done]
    skipped_done = [r for r in records if r in done]
    if skipped_done:
        print(f"already in the file, not repeated: {skipped_done}")
    print(f"to extract: {todo}\n(each record is about 8 hours of ECG; the file is saved after every record, so you can stop "
          f"and run the same command again to continue)\n")
    import time
    wait = float(os.environ.get("APNEA_RETRY_WAIT", "10"))          # seconds; only changed by the tests

    class NetworkDown(Exception):
        pass

    def fetch(rec):
        """Download a record and its labels. Retries brief network trouble; a missing record raises at once."""
        for attempt in (1, 2, 3):
            try:
                return wfdb.rdrecord(rec, pn_dir="apnea-ecg"), wfdb.rdann(rec, "apn", pn_dir="apnea-ecg")
            except Exception as e:
                msg = str(e).lower()
                if not any(k in msg for k in ("connection", "resolve", "timed out", "timeout", "max retries", "temporarily", "reset")):
                    raise                                           # e.g. the record does not exist: caller skips it
                if attempt == 3:
                    raise NetworkDown(str(e))
                print(f"  network problem ({type(e).__name__}); trying again in {wait * attempt:.0f} s ...", flush=True)
                time.sleep(wait * attempt)

    net_failures = 0
    not_done = []
    for rec in todo:
        print(f"downloading {rec} ...", flush=True)
        try:
            record, ann = fetch(rec)
            net_failures = 0
        except NetworkDown as e:
            net_failures += 1
            not_done.append(rec)
            print(f"  could not reach PhysioNet for {rec} after 3 tries")
            if net_failures >= 2:
                not_done += [r for r in todo[todo.index(rec) + 1:]]
                print("\nThe internet connection seems to be down, so I am stopping here instead of failing on every record."
                      "\nEverything extracted so far is saved. Fix the connection, then run the SAME command again:"
                      "\nit skips what is done and carries on with the rest.")
                break
            continue
        except Exception as e:
            print(f"  skipped {rec}: {e}")
            continue
        orig_fs = record.fs
        sig = record.p_signal[:, 0].astype(np.float64)
        sig_250 = resample_to_250(sig, orig_fs)
        rows = extract_record(sig_250, ann.sample, ann.symbol, orig_fs, verbose_name=rec)
        n_pos = sum(1 for r in rows if r[0] == 1)
        print(f"  {rec}: {len(rows)} usable minutes, {n_pos} apnea / {len(rows) - n_pos} normal", flush=True)
        for label, feats in rows:
            rows_out.append([rec, label] + feats)
        save()
    if not os.path.exists(out_path):
        save()
    if not_done:
        print(f"\nstill to do ({len(not_done)}): {not_done}")
    n_pos = sum(int(float(r[1])) for r in rows_out)
    print(f"\nwrote {out_path}: {len(rows_out)} rows, {n_pos} apnea / {len(rows_out) - n_pos} normal, "
          f"{len({r[0] for r in rows_out})} records")


if __name__ == "__main__":
    import argparse
    if "--verify" in sys.argv:
        print("Use verify_apnea_ecg_extract.py for the synthetic self-test.")
        sys.exit(1)
    ap = argparse.ArgumentParser(description="Extract per-minute HRV features from PhysioNet Apnea-ECG records.")
    ap.add_argument("--records", nargs="*", default=None,
                    help="records to extract, e.g. a06-a20 c06-c10 (default: the original 15)")
    ap.add_argument("--out", default="apnea_ecg_features.csv")
    ap.add_argument("--append", action="store_true",
                    help="keep the rows already in the file and add only records that are not in it yet")
    ap.add_argument("--overwrite", action="store_true", help="rebuild the file from scratch")
    ap.add_argument("--redo", action="store_true", help="with --append: extract the listed records again even if present")
    a = ap.parse_args()
    recs = expand_records(a.records) if a.records else RECORDS
    run_real(recs, a.out, "append" if a.append else ("overwrite" if a.overwrite else "new"), a.redo)
