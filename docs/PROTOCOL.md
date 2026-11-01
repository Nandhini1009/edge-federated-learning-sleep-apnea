# FedApnea node <-> aggregator protocol (v1)

Everything the ESP32 firmware must reproduce. All numbers are **little-endian**
(the ESP32 is little-endian, so `memcpy` into a packed struct works).

## Model (see `spec.py`)

8 HRV features -> Dense(16)+ReLU -> Dense(8)+ReLU  **[BASE, shared]**
                -> Dense(1)+Sigmoid                **[HEAD, private]**

| Part  | Params | float32 bytes |
|-------|--------|---------------|
| BASE  | 280    | 1120          |
| HEAD  | 9      | 36            |
| Total | 289    | 1156          |

Parameter vector layout, layer by layer: `W` (out x in, row-major, `W[o*in + i]`)
then `b` (out). Order: layer1 W, layer1 b, layer2 W, layer2 b, head W, head b.

- **FedPer** (default): shared vector = first 280 floats (base only).
- **FedAvg** (baseline): shared vector = all 289 floats. Start the server with `--mode fedavg`.
  The node just slices `params[:n_params]` using the `n_params` in the download header.

Training on the node: SGD, batch 32, learning rate 0.05, 3 epochs per round, weighted
binary cross-entropy (`pos_weight = n_negative / n_positive` in the local train set),
gradient of the loss w.r.t. the output pre-activation:
`dz = p*(1-y) - pos_weight*y*(1-p)`. Test set = last 30 % of the node's windows
(time-ordered, never shuffled). Score = balanced accuracy on that test set.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/round`  | `{"round": N}`. Cheap poll: has the round advanced? |
| GET  | `/model?node=ID` | Global shared weights (binary, below) |
| POST | `/update?node=ID` | Upload locally trained shared weights (binary, below) |
| GET  | `/status` | JSON: mode, round, per-round history (score, bytes) |
| POST | `/reset`  | Start a new run from round 0 |

### Download: `GET /model`  (10-byte header + weights)

| Offset | Type | Field |
|--------|------|-------|
| 0  | char[4]  | magic `"FLM1"` |
| 4  | uint32   | round |
| 8  | uint16   | n_params (280 for FedPer) |
| 10 | float32[n_params] | shared weights |

Total for FedPer: 10 + 1120 = **1130 bytes**.

### Upload: `POST /update`  (18-byte header + weights)

| Offset | Type | Field |
|--------|------|-------|
| 0  | char[4]  | magic `"FLU1"` |
| 4  | uint32   | round (must equal the server's current round, else HTTP 409) |
| 8  | uint32   | n_samples (size of the node's TRAIN set; used as the averaging weight) |
| 12 | float32  | score (balanced accuracy on the held-out test set) |
| 16 | uint16   | n_params (must equal the download's n_params) |
| 18 | float32[n_params] | locally trained shared weights |

Total for FedPer: 18 + 1120 = **1138 bytes**. Response: JSON, e.g.
`{"status":"ok","received":3,"needed":5,"round":7}`.
Errors: 400 (bad format / non-finite values), 409 (`stale_round`).

## Node loop (what the ESP32 does each round)

1. `GET /round`, remember it as `r`.
2. `GET /model` -> copy the weights into `params[0 .. n_params)`; the head (if FedPer) stays as is.
3. Train E epochs on the train windows, evaluate on the test windows.
4. `POST /update` with round `r`.
5. Poll `GET /round` every ~0.5-1 s until it is greater than `r`, then go to step 1.

## C++ for the ESP32

Implemented in `firmware/esp32_fl_node*/`. The header layout it uses:

```cpp
#pragma pack(push, 1)
struct ModelHdr  { char magic[4]; uint32_t round; uint16_t n_params; };                       // 10 bytes
struct UpdateHdr { char magic[4]; uint32_t round; uint32_t n_samples; float score; uint16_t n_params; }; // 18 bytes
#pragma pack(pop)
static_assert(sizeof(ModelHdr) == 10, "");
static_assert(sizeof(UpdateHdr) == 18, "");

// Download: HTTPClient http; http.begin(url + "/model?node=esp32-01"); http.GET();
//   WiFiClient* s = http.getStreamPtr();  s->readBytes(buf, len);   // binary: use the stream, not getString()
// Upload:   http.begin(url + "/update?node=esp32-01");
//   http.addHeader("Content-Type", "application/octet-stream");  http.POST(buf, len);
```

## Feature list

`mean_rr_s, sdnn_s, rmssd_s, pnn50, cv_rr, rr_range_s, median_rr_s, mean_abs_diff_s`,
computed per 60 s window of RR intervals, after rejecting RR outside 0.3-2.0 s.
The per-feature z-score constants (mean/std) are stored in `model_export.h` next to the weights,
so the board and the Pi normalise identically.
