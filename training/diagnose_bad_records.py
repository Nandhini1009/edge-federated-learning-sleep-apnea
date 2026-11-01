#!/usr/bin/env python3
"""
Diagnostic v2 for records that gave 0 or near-0 usable minutes.

v1 checked for NaN - result: ZERO NaN in any of a03/b04/c01/c02/c04/c05, so
that theory is disproven for these specific records (the NaN-poisoning bug
it found is real and still worth having fixed, just isn't what's happening
here). New theory, based on all six records clipping at an almost identical
+-10.2ish bound: ADC saturation. This version checks that directly, and also
runs the actual detector on both a broken and a working record so we can see
concretely whether it's detecting almost nothing (threshold too high for a
weak/clipped signal) or a flood of implausible beats (threshold miscalibrated
low), rather than guessing further.

Run on your laptop: python diagnose_bad_records.py
"""
import numpy as np
import wfdb

from apnea_ecg_extract import resample_to_250, detect_r_peaks_causal, FS, RR_MIN_S, RR_MAX_S

BAD = ["a03", "b04", "c01", "c02", "c04", "c05"]
GOOD_REFERENCE = "a01"   # a record that extracted cleanly, for comparison

for rec in BAD + [GOOD_REFERENCE]:
    print(f"\n=== {rec} ===")
    try:
        r = wfdb.rdrecord(rec, pn_dir="apnea-ecg")
    except Exception as e:
        print("  rdrecord failed:", e)
        continue
    sig = r.p_signal[:, 0]
    print(f"  fs={r.fs}  sig_len={r.sig_len}  duration={r.sig_len/r.fs/60:.1f} min")
    print(f"  NaN count: {np.isnan(sig).sum()} / {len(sig)}  "
          f"({100*np.isnan(sig).sum()/len(sig):.1f}%)")
    finite = sig[np.isfinite(sig)]
    if len(finite):
        print(f"  finite range: min={finite.min():.4f} max={finite.max():.4f} std={finite.std():.4f}")
    # find the longest contiguous finite stretch, AND the longest contiguous
    # bad (NaN) stretch, both in minutes - this tells us whether the fixed
    # extraction script's 2-second interpolation threshold was correctly too
    # short to bridge a real multi-minute dropout (expected, correct
    # exclusion) versus something else going on.
    isnan = np.isnan(sig)
    finite_run = 0; best_finite = 0
    bad_run = 0; best_bad = 0
    for v in isnan:
        if v:
            bad_run += 1; finite_run = 0
        else:
            finite_run += 1; bad_run = 0
        best_finite = max(best_finite, finite_run)
        best_bad = max(best_bad, bad_run)
    print(f"  longest clean (non-NaN) stretch: {best_finite/r.fs/60:.1f} min")
    print(f"  longest bad (NaN) stretch:       {best_bad/r.fs/60:.1f} min"
          f"  ({'<=' if best_bad/r.fs <= 2.0 else '>'} the 2s interpolation limit)")

    # ---- clipping / saturation check ----
    if len(finite):
        clip_level = np.abs(finite).max()
        near_clip = (np.abs(finite) > 0.99 * clip_level).sum()
        print(f"  fraction of samples within 1% of the max abs value "
              f"({clip_level:.4f}): {100*near_clip/len(finite):.2f}%")
        # a flat/stuck signal (repeated identical consecutive values) is a
        # different but related artifact - also worth ruling out
        flat_run = 0; best_flat = 0
        prev = None
        for v in finite:
            if prev is not None and v == prev:
                flat_run += 1
            else:
                flat_run = 0
            best_flat = max(best_flat, flat_run)
            prev = v
        print(f"  longest run of an EXACTLY repeated value: "
              f"{best_flat/r.fs:.1f}s ({best_flat} samples)")

    # ---- run the actual detector, same as extraction does ----
    sig_250 = resample_to_250(sig, r.fs)
    peaks = detect_r_peaks_causal(sig_250)
    dur_hr = len(sig) / r.fs / 3600.0
    implied_hr = len(peaks) / dur_hr / 60.0 if dur_hr > 0 else 0
    print(f"  detector found {len(peaks)} peaks over {dur_hr*60:.0f} min "
          f"-> implied average HR = {implied_hr:.0f} bpm "
          f"({'looks like noise/flood' if implied_hr > 200 else 'looks like under-detection' if implied_hr < 30 else 'plausible'})")
    if len(peaks) > 1:
        rr = np.diff(peaks) / FS
        in_range = ((rr >= RR_MIN_S) & (rr <= RR_MAX_S)).sum()
        print(f"  of {len(rr)} RR intervals, {in_range} ({100*in_range/len(rr):.1f}%) "
              f"are in the physiologically plausible {RR_MIN_S}-{RR_MAX_S}s range")
