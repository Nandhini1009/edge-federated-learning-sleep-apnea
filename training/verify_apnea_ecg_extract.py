"""
Self-test for apnea_ecg_extract.py's pipeline (resample -> detect -> window
-> label), using a SYNTHETIC record with known injected ground truth, since
this sandbox has no internet to fetch the real PhysioNet database. Run this
once here; the real script (apnea_ecg_extract.py) reuses the exact same
detect_r_peaks_causal / hrv_features / extract_record functions unchanged.
"""
import numpy as np
from apnea_ecg_extract import resample_to_250, extract_record, FS

ORIG_FS = 100.0     # PhysioNet apnea-ecg's native rate
N_MIN = 20          # synthetic record length in minutes


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
    def uniform(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) * (1.0 / 16777216.0)


def make_synthetic_record(seed=7):
    """
    N_MIN minutes at ORIG_FS Hz. Alternates apnea/normal minutes with
    different heart-rate statistics (apnea minutes: slower, more variable -
    a simplified stand-in for the real physiological pattern), so the
    pipeline has two genuinely distinct classes to recover.
    """
    lcg = Lcg(seed)
    dur_s = N_MIN * 60
    n = int(dur_s * ORIG_FS)
    t = np.arange(n) / ORIG_FS
    sig = np.zeros(n)

    minute_label = np.array(['A' if (m % 3 == 1) else 'N' for m in range(N_MIN)])  # 1 in 3 apnea
    beat_times = []
    tb = 0.3
    while tb < dur_s - 0.3:
        minute = int(tb // 60)
        is_apnea = minute_label[minute] == 'A'
        mean_hr = 55.0 if is_apnea else 72.0
        jitter = 12.0 if is_apnea else 5.0
        beat_times.append(tb)
        rr = 60.0 / (mean_hr + (lcg.uniform() * 2 - 1) * jitter)
        tb += rr
    beat_times = np.array(beat_times)

    def bump(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    for bt in beat_times:
        sig += bump(bt - 0.16, 0.025, 0.15)
        sig += bump(bt, 0.012, 1.0)
        sig += bump(bt + 0.28, 0.05, 0.25)
    noise = np.array([(lcg.uniform() * 2 - 1) for _ in range(n)]) * 0.03
    sig += noise

    ann_sample = np.array([m * 60 * ORIG_FS for m in range(N_MIN)], dtype=np.int64)
    return sig.astype(np.float32), ann_sample, minute_label, beat_times, minute_label


sig, ann_sample, ann_label, true_beats, true_minute_label = make_synthetic_record()
print(f"synthetic record: {N_MIN} min @ {ORIG_FS} Hz native, {len(true_beats)} true beats")
print(f"injected labels: {list(ann_label)}")

sig_250 = resample_to_250(sig, ORIG_FS)
print(f"resampled {len(sig)} samples @ {ORIG_FS}Hz -> {len(sig_250)} samples @ {FS}Hz")

rows = extract_record(sig_250, ann_sample, ann_label, ORIG_FS, min_rr_per_window=10)
print(f"\nextracted {len(rows)} labeled windows (of {N_MIN} minutes, "
      f"first ~2 min lost to detector calibration + any minute with too few RR)")

correct_class_recovered = 0
for i, (label, feats) in enumerate(rows):
    mean_rr = feats[0]
    print(f"  row {i}: label={label} ({'apnea' if label else 'normal'})  "
          f"mean_rr={mean_rr:.3f}s ({60/mean_rr:.0f} bpm)  sdnn={feats[1]:.3f}")

# sanity: apnea-labeled windows should show a visibly slower, more variable
# heart rate than normal-labeled windows, since that's what we injected
apnea_hr = [60.0 / f[0] for l, f in rows if l == 1]
normal_hr = [60.0 / f[0] for l, f in rows if l == 0]
print(f"\nmean HR: apnea-labeled windows={np.mean(apnea_hr):.1f} bpm  "
      f"normal-labeled windows={np.mean(normal_hr):.1f} bpm")
assert np.mean(apnea_hr) < np.mean(normal_hr) - 5, \
    "the extracted features don't separate the two injected classes - pipeline bug"
print("PASS: the pipeline correctly recovers the injected HR difference between classes.")
