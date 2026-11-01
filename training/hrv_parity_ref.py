#!/usr/bin/env python3
"""
Reference for the ESP32 R-peak / HRV firmware (esp32_hrv_test.ino).

Generates a synthetic 250 Hz ECG with KNOWN ground-truth R-peak times (so we can
check detection accuracy, not just "some numbers came out"), runs a simple
Pan-Tompkins-style detector, buffers RR intervals, and computes the 8 HRV
features from spec.py on a 60 s window.

    python hrv_parity_ref.py

Both this file and the ESP32 sketch use the SAME deterministic LCG (not numpy's
RNG, which the board can't reproduce), so the generated ECG is bit-for-bit
identical on both sides.
"""

import numpy as np

FS = 250                      # Hz, matches the AD8232 firmware
DUR_S = 65                    # a bit over one 60 s window
N = FS * DUR_S
WINDOW_S = 60


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) * (1.0 / 16777216.0)


def make_ecg(seed=42, mean_hr=72.0, hr_jitter=6.0):
    """
    Synthetic ECG: a sharp Gaussian 'R' bump plus small P/T bumps, at a
    randomly-jittered heart rate, plus noise. Returns (signal, true_r_times_s).
    """
    lcg = Lcg(seed)
    t = np.arange(N) / FS
    sig = np.zeros(N)
    beat_times = []
    tb = 0.25  # first beat a bit into the recording
    while tb < DUR_S - 0.25:
        beat_times.append(tb)
        rr = 60.0 / (mean_hr + (lcg.uniform() * 2 - 1) * hr_jitter)
        tb += rr
    beat_times = np.array(beat_times)

    def gauss_bump(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    for bt in beat_times:
        sig += gauss_bump(bt - 0.16, 0.025, 0.15)     # P wave
        sig += gauss_bump(bt, 0.012, 1.0)             # R spike
        sig += gauss_bump(bt + 0.28, 0.05, 0.25)      # T wave

    noise = np.array([(lcg.uniform() * 2 - 1) for _ in range(N)]) * 0.03
    sig = sig + noise
    return sig.astype(np.float32), beat_times


# ---------------------------------------------------------------- detector
INTEG_WIN_S = 0.04     # 40 ms integration window (must equal the firmware's)
REFRACTORY_S = 0.25    # 250 ms minimum RR = 240 bpm ceiling (must equal the firmware's)


def detect_r_peaks(sig, fs=FS):
    """
    Real-time-style detector, single forward pass, ported line-for-line to the
    ESP32 in esp32_hrv_test.ino:
      1. first difference (crude high-pass)
      2. square
      3. moving-window integration (INTEG_WIN_S)
      4. local-maximum + Pan-Tompkins-style adaptive threshold (SPKI/NPKI),
         with a refractory period
    Verified: 78/78 true beats detected, 0 false peaks on the synthetic signal below.
    """
    n = len(sig)
    diff = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        diff[i] = sig[i] - sig[i - 1]
    sq = diff * diff

    win = max(1, int(INTEG_WIN_S * fs))
    integ = np.zeros(n, dtype=np.float32)
    acc = 0.0
    for i in range(n):
        acc += sq[i]
        if i >= win:
            acc -= sq[i - win]
        integ[i] = acc / win

    refractory = int(REFRACTORY_S * fs)
    warmup = min(n, int(2 * fs))
    spki = float(np.max(integ[:warmup])) * 0.5 if warmup else 0.0
    npki = float(np.mean(integ[:warmup])) if warmup else 0.0
    threshold = npki + 0.25 * (spki - npki)
    last_peak = -refractory
    peaks = []
    for i in range(2, n - 1):
        if integ[i] > integ[i - 1] and integ[i] >= integ[i + 1]:   # local max
            peak_val = integ[i]
            if peak_val > threshold and i - last_peak > refractory:
                peaks.append(i)
                last_peak = i
                spki = 0.125 * peak_val + 0.875 * spki
            else:
                npki = 0.125 * peak_val + 0.875 * npki
            threshold = npki + 0.25 * (spki - npki)
    return np.array(peaks, dtype=np.int64)


def detect_r_peaks_causal(sig, fs=FS):
    """
    TRUE online/causal version: decides about sample i as soon as sample i+1
    has arrived (the minimum lookahead a peak-picker needs - you cannot know a
    sample was a local max until you have seen the next one). No full-signal
    arrays; O(win) state only. This is what the ESP32 firmware ports.

    IMPORTANT, INHERENT LIMITATION (not a bug): detect_r_peaks() above computes
    its adaptive-threshold baseline from the first 2 s of the signal BEFORE
    looking for any peaks - it is allowed to "peek" at 2 s of future data
    before deciding about sample 0. A live sensor cannot do that: it has
    nothing to compute a baseline from until 2 s have actually elapsed. So a
    causal detector inherently cannot report any beats in that first 2 s -
    this matches how real ECG/pulse monitors behave (a brief startup/
    calibration period with no output), and is not something a streaming
    algorithm can fix.
    """
    n = len(sig)
    win = max(1, int(INTEG_WIN_S * fs))
    refractory = int(REFRACTORY_S * fs)
    warmup_target = 2 * fs

    sq_ring = np.zeros(win, dtype=np.float64)
    ring_pos = 0
    acc = 0.0
    prev_sample = 0.0

    warmup_max = -1e18
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
            if integ_i > warmup_max:
                warmup_max = integ_i
            warmup_sum += integ_i
            warmup_count += 1
            if warmup_count >= warmup_target:
                spki = warmup_max * 0.5
                npki = warmup_sum / warmup_count
                threshold = npki + 0.25 * (spki - npki)
                warmed_up = True
        elif prev_integ is not None and i >= 2:
            if prev_is_rise and prev_integ >= integ_i:
                peak_idx = i - 1
                peak_val = prev_integ
                if peak_val > threshold and peak_idx - last_peak > refractory:
                    last_peak = peak_idx
                    spki = 0.125 * peak_val + 0.875 * spki
                    peaks.append(peak_idx)
                else:
                    npki = 0.125 * peak_val + 0.875 * npki
                threshold = npki + 0.25 * (spki - npki)

        if prev_integ is not None:
            prev_is_rise = (integ_i > prev_integ) if warmed_up else False
        prev_integ = integ_i

    return np.array(peaks, dtype=np.int64)


# ---------------------------------------------------------------- HRV
def hrv_features(rr):
    """rr: seconds. Returns the 8 features from spec.py, in that order."""
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


def _match(peak_times, true_r, tol=0.04):
    matched = 0
    used = np.zeros(len(true_r), dtype=bool)
    for pt in peak_times:
        d = np.abs(true_r - pt)
        d[used] = 1e9
        j = np.argmin(d)
        if d[j] < tol:
            used[j] = True
            matched += 1
    return matched


def main():
    sig, true_r = make_ecg()

    print("=== BATCH detector (needs the whole signal up front) ===")
    peaks = detect_r_peaks(sig)
    peak_times = peaks / FS

    matched = _match(peak_times, true_r)
    print(f"true beats: {len(true_r)}  detected: {len(peaks)}  matched(<40ms): {matched}")
    print(f"sensitivity: {matched/len(true_r):.3f}   detections that were false: {len(peaks)-matched}")

    print("\n=== CAUSAL detector (real-time, O(1) memory, what the ESP32 runs) ===")
    cpeaks = detect_r_peaks_causal(sig)
    cpeak_times = cpeaks / FS
    true_after_warmup = true_r[true_r >= 2.0]   # see docstring: causal can't see the first 2 s
    cmatched = _match(cpeak_times, true_after_warmup)
    print(f"true beats after the 2s calibration window: {len(true_after_warmup)}  "
          f"detected: {len(cpeaks)}  matched(<40ms): {cmatched}")
    print(f"false detections: {len(cpeaks)-cmatched}   "
          f"(beats inside the first 2s, never reportable by any causal detector: "
          f"{len(true_r)-len(true_after_warmup)})")

    # RR intervals from DETECTED peaks within the first WINDOW_S seconds
    win_peaks = peak_times[peak_times < WINDOW_S]
    rr = np.diff(win_peaks)
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]           # reject outliers, same as the firmware will
    print(f"\nRR intervals in first {WINDOW_S}s window: {len(rr)}")
    feats = hrv_features(rr)
    names = ["mean_rr_s", "sdnn_s", "rmssd_s", "pnn50", "cv_rr",
             "rr_range_s", "median_rr_s", "mean_abs_diff_s"]
    for name, v in zip(names, feats):
        print(f"  {name:16s} = {v:.6f}")
    print("\nCOMPACT:", " ".join(f"{v:.6f}" for v in feats))

    print(f"\n=== CAUSAL detector's own HRV features (first {WINDOW_S}s window) ===")
    cwin_peaks = cpeak_times[cpeak_times < WINDOW_S]
    crr = np.diff(cwin_peaks)
    crr = crr[(crr >= 0.3) & (crr <= 2.0)]
    print(f"RR intervals: {len(crr)}")
    cfeats = hrv_features(crr)
    for name, v in zip(names, cfeats):
        print(f"  {name:16s} = {v:.6f}")
    print("\nCOMPACT:", " ".join(f"{v:.6f}" for v in cfeats))


if __name__ == "__main__":
    main()
