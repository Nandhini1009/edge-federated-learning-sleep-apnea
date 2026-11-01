"""
FedApnea shared model spec.

The server, the simulated nodes and (later) the ESP32 firmware must all agree on
this layout EXACTLY. If you change anything here, change the firmware too.

Model = small MLP on 8 HRV features, one output (apnea probability).

    BASE  (shared, federated):  Dense(8 -> 16) + ReLU, Dense(16 -> 8) + ReLU
    HEAD  (private, local)   :  Dense(8 -> 1)  + Sigmoid

Parameter vector layout (float32, little-endian), layer by layer:
    for each layer:  W (out x in, row-major: W[o*in + i])  then  b (out)
Base params come first, head params last, so:
    FedPer :  shared vector = params[:N_BASE]
    FedAvg :  shared vector = params[:N_TOTAL]   (head is averaged too)
"""

import numpy as np

# Final feature list (per 60 s window of RR intervals). Do not change it without retraining:
# the board, the Pi and the trained weights all depend on this exact order.
FEATURES = [
    "mean_rr_s",        # mean RR interval (s)
    "sdnn_s",           # std of RR intervals (s)
    "rmssd_s",          # root mean square of successive RR differences (s)
    "pnn50",            # fraction of successive differences > 50 ms
    "cv_rr",            # sdnn / mean_rr
    "rr_range_s",       # max RR - min RR (s)
    "median_rr_s",      # median RR (s)
    "mean_abs_diff_s",  # mean |RR[i+1] - RR[i]| (s)
]
N_FEAT = len(FEATURES)

# (in, out) per layer. The LAST layer is the personalised head.
LAYERS = [(N_FEAT, 16), (16, 8), (8, 1)]
N_BASE_LAYERS = 2


def _count(layers):
    return sum(i * o + o for i, o in layers)


N_BASE = _count(LAYERS[:N_BASE_LAYERS])   # 280
N_TOTAL = _count(LAYERS)                   # 289


def init_params(rng):
    """He-initialised full parameter vector (float32)."""
    parts = []
    for i, o in LAYERS:
        W = rng.normal(0.0, np.sqrt(2.0 / i), size=(o, i))
        parts.append(W.reshape(-1))
        parts.append(np.zeros(o))
    return np.concatenate(parts).astype(np.float32)


def unpack(vec):
    """Return [(W, b), ...] as VIEWS into vec, so in-place updates modify vec."""
    out, pos = [], 0
    for i, o in LAYERS:
        W = vec[pos:pos + o * i].reshape(o, i)
        pos += o * i
        b = vec[pos:pos + o]
        pos += o
        out.append((W, b))
    return out


if __name__ == "__main__":
    print("features :", N_FEAT)
    print("N_BASE   :", N_BASE, "params =", N_BASE * 4, "bytes (float32)")
    print("N_TOTAL  :", N_TOTAL, "params =", N_TOTAL * 4, "bytes (float32)")
