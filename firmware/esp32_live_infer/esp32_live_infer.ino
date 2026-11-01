/*
  FedApnea ESP32 - LIVE sensor -> the same model Python validated
  ----------------------------------------------------------------------------
  Every step copies what apnea_ecg_extract.py did to the PhysioNet recordings,
  so a live minute of ECG becomes the same kind of 8 numbers the model was
  trained on:

    AD8232 ADC (250 Hz) -> 50 Hz notch -> beat detector -> RR intervals
      -> last 60 s of RR (0.3-2.0 s only, at least 20 of them)
      -> 8 HRV features -> scaled with FEAT_MU/FEAT_SD -> model -> probability

  The beat detector is the Python one, including its two robustness fixes
  (95th-percentile warm-up, cap on how much one spike can raise the threshold),
  and it runs in double precision so it makes the same decisions as Python.
  A lead-off or a flat/stuck signal (> 2 s of one identical value) restarts the
  detector and invalidates any window that contains it - same as the extraction.

  Before using this sketch, copy model_export.h (made by
  export_model_for_esp32.py, the one esp32_model_validate checked) into THIS folder.

  Wiring: AD8232 OUTPUT->GPIO34, LO+->GPIO32, LO-->GPIO33, 3.3V, GND. Three
  electrodes on the body. Serial Monitor 115200.

  READ THIS: the model scores about 0.65-0.68 balanced accuracy on held-out
  patients and your own ECG has no apnea labels. A probability here is not a
  diagnosis. What this sketch proves is that the sensor path produces the same
  numbers Python would produce from the same signal.
*/

#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if !__has_include("model_export.h")
#error "Copy model_export.h (from the esp32_model_validate folder) into this sketch folder."
#endif
#include "model_export.h"

// 0 = live sensor (normal use).  1 = REPLAY: the laptop (replay_validate.py) streams real
// PhysioNet ECG samples over USB into the very same pipeline below, so the board's beats,
// features and probabilities can be compared with Python's on real recordings.
#ifndef REPLAY_MODE
#define REPLAY_MODE 0
#endif

// 1 = also send one small report every 10 s to the Raspberry Pi (dashboard_server.py). The Pi runs the
// model, takes the decision and shows it on a web dashboard. Its reply comes back here: the board prints the
// decision on Serial and confirms in its next report what it received.
// Only the 8 summary numbers and signal-quality values leave the board, never the ECG waveform.
// The sensor loop never waits for the network: Wi-Fi/HTTP run in their own task on the other CPU core.
// Set it to 0 to run fully offline, exactly as before.
#ifndef DASHBOARD
#define DASHBOARD 1
#endif

// 0 (default) = private mode: only the 8 summary numbers and signal-quality values leave the board.
// 1 = each 10 s report also carries a raw-ECG PREVIEW (the last 10 s, 125 samples per second) so the
//     dashboard can draw the waveform. It is for DISPLAY only: the Pi shows it, never stores it and never uses it for learning.
#ifndef ECG_STREAM
#define ECG_STREAM 0
#endif
#define ECG_DIV 2          // keep every 2nd sample: 250 Hz -> 125 Hz
#define ECG_MAX 1300       // 10 s x 125 Hz = 1250, plus a little room

#if DASHBOARD && !REPLAY_MODE
#include <WiFi.h>
#include <HTTPClient.h>
// ============================ EDIT THESE ============================
static const char* WIFI_SSID = "YOUR_WIFI_NAME";
static const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
static const char* DASH_HOST = "YOUR_PI_IP";        // the Raspberry Pi that runs dashboard_server.py
static const int   DASH_PORT = 8081;
static const char* DEVICE_ID = "esp32-live";
// ====================================================================
struct Report {            // what one 10 s report carries (the board fills it, the network task sends it)
  long  tSec;
  int   valid, nRR, span;
  long  beats10, leadoff;
  float cov, boardProb;
  float feat[8];
  char  why[64];
#if ECG_STREAM
  int      ecgN;
  uint16_t ecg[ECG_MAX];       // raw ADC counts (0-4095), 125 samples per second
#endif
};
#endif

#define ECG_PIN   34
#define LO_PLUS   32
#define LO_MINUS  33

#define FS            250
#define WINDOW_S      60        // one feature window = the last 60 s, like one annotated minute
#define PRED_PERIOD_S 10        // how often to produce a prediction
#define MIN_RR        20        // same minimum as the extraction
#define COVER_MIN     0.70      // live safety rule (NOT from the training data): the valid beat-to-beat
                                // intervals must add up to at least 70% of the 60 s window, otherwise
                                // most of the minute went undetected and the features mean nothing
#define RR_MIN_S      0.3
#define RR_MAX_S      2.0

// detector constants - identical to apnea_ecg_extract.py at 250 Hz
#define INTEG_WIN      10       // int(0.04 * 250)
#define REFRACTORY     62       // int(0.25 * 250)
#define WARMUP_N       500      // 2 s
#define STUCK_SAMPLES  501      // Python: a run of MORE than 2.0 s (500 samples) of one value
static const double SPKI_MAX_JUMP = 5.0;

static const uint32_t ECG_PERIOD_US = 1000000UL / FS;

// ---------------- 50 Hz notch for the AD8232 (unchanged, verified earlier) ----------------
struct Notch {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
  float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  void init(float f0, float fs, float Q) {
    float w0 = 2.0f * PI * f0 / fs;
    float alpha = sinf(w0) / (2.0f * Q);
    float c = cosf(w0);
    float a0 = 1.0f + alpha;
    b0 = 1.0f / a0; b1 = -2.0f * c / a0; b2 = 1.0f / a0;
    a1 = -2.0f * c / a0; a2 = (1.0f - alpha) / a0;
  }
  void reset() { x1 = x2 = y1 = y2 = 0; }
  float process(float x) {
    float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x; y2 = y1; y1 = y;
    return y;
  }
};
static Notch notch50;

// ---------------- beat detector: apnea_ecg_extract.py's detect_r_peaks_causal ----------------
struct Detector {
  double sqRing[INTEG_WIN];
  int    ringPos;
  double acc, prevSample;
  double warmupVals[WARMUP_N];
  double warmupSum;
  int    warmupCount;
  bool   warmedUp;
  double spki, npki, threshold;
  bool   havePrevInteg;
  double prevInteg;
  bool   prevIsRise;
  long   lastPeak;
  long   i;

  void begin() {
    for (int k = 0; k < INTEG_WIN; k++) sqRing[k] = 0.0;
    ringPos = 0; acc = 0.0; prevSample = 0.0;
    warmupSum = 0.0; warmupCount = 0; warmedUp = false;
    spki = npki = threshold = 0.0;
    havePrevInteg = false; prevInteg = 0.0; prevIsRise = false;
    lastPeak = -REFRACTORY; i = 0;
  }

  static int cmpD(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
  }

  // Feed one sample. Returns true when a beat was accepted; *peakIdx is the
  // beat's index within this segment (it is decided one sample late).
  bool step(double x, long* peakIdx) {
    long idx = i++;
    double diff = (idx == 0) ? 0.0 : (x - prevSample);
    prevSample = x;
    double sq = diff * diff;
    acc += sq;
    if (idx >= INTEG_WIN) acc -= sqRing[ringPos];
    sqRing[ringPos] = sq;
    ringPos = (ringPos + 1) % INTEG_WIN;
    double integ = acc / INTEG_WIN;

    bool found = false;
    if (!warmedUp) {
      if (warmupCount < WARMUP_N) warmupVals[warmupCount] = integ;
      warmupSum += integ;
      warmupCount++;
      if (warmupCount >= WARMUP_N) {
        // 95th percentile with linear interpolation = numpy.percentile(vals, 95)
        qsort(warmupVals, WARMUP_N, sizeof(double), cmpD);
        double virt = (WARMUP_N - 1) * 0.95;
        long lo = (long)floor(virt);
        double gamma = virt - (double)lo;
        double a = warmupVals[lo];
        double b = warmupVals[(lo + 1 < WARMUP_N) ? lo + 1 : lo];
        double robustMax = (gamma >= 0.5) ? (b - (b - a) * (1.0 - gamma)) : (a + (b - a) * gamma);
        spki = robustMax * 0.5;
        npki = warmupSum / warmupCount;
        threshold = npki + 0.25 * (spki - npki);
        warmedUp = true;
      }
    } else if (havePrevInteg && idx >= 2) {
      if (prevIsRise && prevInteg >= integ) {
        long pk = idx - 1;
        double pv = prevInteg;
        if (pv > threshold && pk - lastPeak > REFRACTORY) {
          lastPeak = pk;
          double capped = (spki > 0.0) ? fmin(pv, SPKI_MAX_JUMP * spki) : pv;
          spki = 0.125 * capped + 0.875 * spki;
          *peakIdx = pk;
          found = true;
        } else {
          npki = 0.125 * pv + 0.875 * npki;
        }
        threshold = npki + 0.25 * (spki - npki);
      }
    }
    if (havePrevInteg) prevIsRise = warmedUp ? (integ > prevInteg) : false;
    prevInteg = integ;
    havePrevInteg = true;
    return found;
  }
};
static Detector det;

// ---------------- pipeline state ----------------
static long gSample = 0;          // global index of the next sample
static long segStart = 0;         // global index where the current detector segment began
static bool needsBegin = true;
static bool havePeak = false;
static long lastPeakG = 0;
static long lastBadG = -1000000;  // last sample that was lead-off or part of a stuck run
static long prevKey = 0;
static bool haveKey = false;
static long eqRun = 0;
static bool inStuck = false;
static long beatsTotal = 0;     // every beat the detector has accepted (diagnostics)

#define RR_CAP 256
static long   rrLater[RR_CAP];    // global index of the LATER beat of each RR interval
static double rrVal[RR_CAP];
static int rrHead = 0, rrCount = 0;
static double rrWin[RR_CAP];
static double rrTmp[RR_CAP];

__attribute__((unused)) static void pipelineResetAll() {
  gSample = 0; segStart = 0; needsBegin = true; havePeak = false; lastPeakG = 0;
  lastBadG = -1000000; haveKey = false; prevKey = 0; eqRun = 0; inStuck = false;
  beatsTotal = 0; rrHead = 0; rrCount = 0;
}

static void markBad() {
  lastBadG = gSample;
  needsBegin = true;
  havePeak = false;
}

static void pipelineLeadOff() {
  markBad();
  haveKey = false;
  eqRun = 0;
  inStuck = false;
  gSample++;
}

// x = notch-filtered sample fed to the detector; rawKey = the raw ADC value,
// used only to spot a stuck (flat) signal.
static void pipelineSample(double x, long rawKey) {
  if (haveKey && rawKey == prevKey) eqRun++; else { eqRun = 1; inStuck = false; }
  prevKey = rawKey;
  haveKey = true;
  if (eqRun >= STUCK_SAMPLES) inStuck = true;
  if (inStuck) { markBad(); gSample++; return; }

  if (needsBegin) {
    det.begin();
    segStart = gSample;
    needsBegin = false;
    havePeak = false;
  }
  long pk;
  if (det.step(x, &pk)) {
    long g = segStart + pk;
    if (havePeak) {
      rrLater[rrHead] = g;
      rrVal[rrHead] = (double)g / FS - (double)lastPeakG / FS;
      rrHead = (rrHead + 1) % RR_CAP;
      if (rrCount < RR_CAP) rrCount++;
    }
    lastPeakG = g;
    havePeak = true;
    beatsTotal++;
  }
  gSample++;
}

// ---------------- 8 HRV features, same formulas as hrv_features() in Python ----------------
static void hrvFeatures(const double* rr, int n, float out[8]) {
  double sum = 0.0, mn = rr[0], mx = rr[0];
  for (int i = 0; i < n; i++) { sum += rr[i]; if (rr[i] < mn) mn = rr[i]; if (rr[i] > mx) mx = rr[i]; }
  double mean = sum / n;
  double var = 0.0;
  for (int i = 0; i < n; i++) { double d = rr[i] - mean; var += d * d; }
  double sdnn = sqrt(var / n);
  int nd = n - 1;
  double sq = 0.0, absSum = 0.0; int over = 0;
  for (int i = 0; i < nd; i++) {
    double d = rr[i + 1] - rr[i];
    sq += d * d; absSum += fabs(d);
    if (fabs(d) > 0.05) over++;
  }
  double rmssd = sqrt(sq / nd);
  double pnn50 = (double)over / nd;
  double mad = absSum / nd;
  double cv = (mean > 0.0) ? sdnn / mean : 0.0;
  memcpy(rrTmp, rr, n * sizeof(double));
  qsort(rrTmp, n, sizeof(double), Detector::cmpD);
  double med = (n % 2 == 1) ? rrTmp[n / 2] : 0.5 * (rrTmp[n / 2 - 1] + rrTmp[n / 2]);
  out[0] = (float)mean; out[1] = (float)sdnn; out[2] = (float)rmssd; out[3] = (float)pnn50;
  out[4] = (float)cv;   out[5] = (float)(mx - mn); out[6] = (float)med; out[7] = (float)mad;
}

// ---------------- the model (same forward pass esp32_model_validate checked) ----------------
#define N_LAYERS 3
static const int L_IN[N_LAYERS]  = {8, 16, 8};
static const int L_OUT[N_LAYERS] = {16, 8, 1};
static int L_W_OFF[N_LAYERS];
static int L_B_OFF[N_LAYERS];
static void mlp_layout() {
  int pos = 0;
  for (int l = 0; l < N_LAYERS; l++) {
    L_W_OFF[l] = pos; pos += L_IN[l] * L_OUT[l];
    L_B_OFF[l] = pos; pos += L_OUT[l];
  }
}
static float mlp_forward(const float* P, const float* x) {
  float a1[16], a2[8];
  for (int o = 0; o < 16; o++) {
    const float* w = P + L_W_OFF[0] + o * 8;
    float z = P[L_B_OFF[0] + o];
    for (int i = 0; i < 8; i++) z += w[i] * x[i];
    a1[o] = z > 0.0f ? z : 0.0f;
  }
  for (int o = 0; o < 8; o++) {
    const float* w = P + L_W_OFF[1] + o * 16;
    float z = P[L_B_OFF[1] + o];
    for (int i = 0; i < 16; i++) z += w[i] * a1[i];
    a2[o] = z > 0.0f ? z : 0.0f;
  }
  const float* w = P + L_W_OFF[2];
  float z = P[L_B_OFF[2]];
  for (int i = 0; i < 8; i++) z += w[i] * a2[i];
  return 1.0f / (1.0f + expf(-z));
}

// Features + probability for the window that ends at endSample (exclusive).
static bool evaluateWindow(long endSample, float feat[8], int* nRR, float* prob, const char** why, double* coverage) {
  *nRR = 0;
  *coverage = 0.0;
  long startSample = endSample - (long)WINDOW_S * FS;
  if (startSample < 0) { *why = "less than 60 s of signal so far"; return false; }
  if (lastBadG >= startSample) { *why = "lead-off or flat signal inside the window"; return false; }
  double t1 = (double)endSample / FS;
  double t0 = t1 - WINDOW_S;
  int idx0 = (rrHead - rrCount + RR_CAP) % RR_CAP;
  if (rrCount == RR_CAP && (double)rrLater[idx0] / FS >= t0) {
    *why = "more than 256 beats in 60 s - far too noisy";
    return false;
  }
  int n = 0;
  for (int k = 0; k < rrCount; k++) {
    int j = (idx0 + k) % RR_CAP;
    double t = (double)rrLater[j] / FS;
    if (t >= t0 && t < t1) {
      double r = rrVal[j];
      if (r >= RR_MIN_S && r <= RR_MAX_S) rrWin[n++] = r;
    }
  }
  *nRR = n;
  double cover = 0.0;
  for (int k = 0; k < n; k++) cover += rrWin[k];
  *coverage = cover / WINDOW_S;
  if (n < MIN_RR) { *why = "fewer than 20 valid beats in the window"; return false; }
  if (*coverage < COVER_MIN) { *why = "the detected beats cover too little of the minute"; return false; }
  hrvFeatures(rrWin, n, feat);
  float xn[8];
  for (int j = 0; j < 8; j++) xn[j] = (feat[j] - FEAT_MU[j]) / FEAT_SD[j];
  *prob = mlp_forward(MODEL_W, xn);
  return true;
}

#if !REPLAY_MODE
#if DASHBOARD
// ---------------- dashboard link: the Pi decides, this board shows it ----------------
static QueueHandle_t reportQ = nullptr;
#if ECG_STREAM
#define BODY_CAP (640 + ECG_MAX * 6)
#else
#define BODY_CAP 640
#endif
static volatile int gAlertLevel = 0;          // the Pi's latest decision: 0 OK, 1 WATCH, 2 ALERT
static char gCrcHex[9];

// CRC-32 (the same one Python's zlib.crc32 computes) over the weights and scaling constants:
// the Pi compares it with its own, so a board running a different model is spotted at once.
static uint32_t crc32Update(uint32_t crc, const uint8_t* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    crc ^= p[i];
    for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc;
}
static void computeModelCrc() {
  uint32_t c = 0xFFFFFFFFu;
  c = crc32Update(c, (const uint8_t*)MODEL_W, sizeof(MODEL_W));
  c = crc32Update(c, (const uint8_t*)FEAT_MU, sizeof(FEAT_MU));
  c = crc32Update(c, (const uint8_t*)FEAT_SD, sizeof(FEAT_SD));
  snprintf(gCrcHex, sizeof(gCrcHex), "%08x", (unsigned)(c ^ 0xFFFFFFFFu));
}

static int buildReportJson(char* b, size_t cap, const Report& r, int rssi, int ack, const char* crc) {
  int n = snprintf(b, cap, "{\"device\":\"%s\",\"t\":%ld,\"valid\":%d,\"n\":%d,\"cov\":%.4f,\"f\":[",
                   DEVICE_ID, r.tSec, r.valid, r.nRR, (double)r.cov);
  for (int j = 0; j < 8 && n > 0 && (size_t)n < cap; j++)
    n += snprintf(b + n, cap - n, "%s%.9g", j ? "," : "", (double)r.feat[j]);
  if (n > 0 && (size_t)n < cap)
    n += snprintf(b + n, cap - n, "],\"bp\":%.9g,\"span\":%d,\"b10\":%ld,\"lo\":%ld,\"rssi\":%d,\"ack\":%d,\"crc\":\"%s\",\"why\":\"%s\"",
                  (double)r.boardProb, r.span, r.beats10, r.leadoff, rssi, ack, crc, r.why);
#if ECG_STREAM
  if (r.ecgN > 0 && n > 0 && (size_t)n < cap)
    n += snprintf(b + n, cap - n, ",\"ecg_hz\":%d,\"ecg\":[", 250 / ECG_DIV);
  for (int j = 0; j < r.ecgN && n > 0 && (size_t)n + 8 < cap; j++)
    n += snprintf(b + n, cap - n, "%s%u", j ? "," : "", (unsigned)r.ecg[j]);
  if (r.ecgN > 0 && n > 0 && (size_t)n < cap)
    n += snprintf(b + n, cap - n, "]");
#endif
  if (n > 0 && (size_t)n < cap) n += snprintf(b + n, cap - n, "}");
  return n;
}

// tiny JSON readers for the Pi's reply  {"level":2,"state":"ALERT","p":0.66,"thr":0.5}
static const char* jsonValue(const char* s, const char* key) {
  char pat[24];
  snprintf(pat, sizeof(pat), "\"%s\":", key);
  const char* p = strstr(s, pat);
  if (!p) return nullptr;
  p += strlen(pat);
  while (*p == ' ') p++;
  return p;
}
static int jsonGetInt(const char* s, const char* key, int def) {
  const char* p = jsonValue(s, key);
  return p ? atoi(p) : def;
}
static void jsonGetString(const char* s, const char* key, char* out, size_t cap) {
  out[0] = 0;
  const char* p = jsonValue(s, key);
  if (!p || *p != '"') return;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
  out[i] = 0;
}

static void maintainWifi() {
  static bool started = false, wasUp = false;
  static uint32_t lastBegin = 0;
  bool up = (WiFi.status() == WL_CONNECTED);
  if (up && !wasUp) Serial.printf("[dashboard] Wi-Fi connected, board address %s\n", WiFi.localIP().toString().c_str());
  wasUp = up;
  if (up) return;
  uint32_t now = millis();
  if (!started || now - lastBegin > 20000) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    started = true;
    lastBegin = now;
    Serial.println("[dashboard] connecting to Wi-Fi ...");
  }
}

// Runs on the OTHER core: slow, blocking network calls happen here, never in the sensor loop.
static void netTask(void*) {
  for (;;) {
    static Report r;               // static: too big for this task's stack
    maintainWifi();
    if (xQueueReceive(reportQ, &r, pdMS_TO_TICKS(1000)) != pdTRUE) continue;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[dashboard] no Wi-Fi - report dropped");
      continue;
    }
    static char body[BODY_CAP];
    int len = buildReportJson(body, sizeof(body), r, WiFi.RSSI(), gAlertLevel, gCrcHex);
    if (len <= 0 || (size_t)len >= sizeof(body)) { Serial.println("[dashboard] report too long - dropped"); continue; }
    HTTPClient http;
    http.begin(String("http://") + DASH_HOST + ":" + String(DASH_PORT) + "/ingest");
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(4000);
    int code = http.POST((uint8_t*)body, (size_t)len);
    if (code == 200) {
      String resp = http.getString();
      int lvl = jsonGetInt(resp.c_str(), "level", -1);
      char state[24];
      jsonGetString(resp.c_str(), "state", state, sizeof(state));
      if (lvl >= 0 && lvl <= 2) {
        int prev = gAlertLevel;
        gAlertLevel = lvl;
        if (lvl == 2 && prev != 2) Serial.printf("[dashboard] *** ALERT from the Pi (%s) ***\n", state);
        else if (prev == 2 && lvl < 2) Serial.printf("[dashboard] alert cleared by the Pi (%s)\n", state);
        else Serial.printf("[dashboard] Pi decision: %s (level %d)\n", state, lvl);
      }
    } else {
      Serial.printf("[dashboard] Pi did not answer (HTTP %d) - is dashboard_server.py running at %s:%d?\n", code, DASH_HOST, DASH_PORT);
    }
    http.end();
  }
}

static void dashboardStart() {
  computeModelCrc();
  reportQ = xQueueCreate(4, sizeof(Report));
  xTaskCreatePinnedToCore(netTask, "dashboard", 10240, nullptr, 1, nullptr, 0);
  Serial.printf("Dashboard link on: reports go to http://%s:%d/ingest (model fingerprint %s)\n", DASH_HOST, DASH_PORT, gCrcHex);
  if (strcmp(WIFI_SSID, "YOUR_WIFI_NAME") == 0)
    Serial.println("[dashboard] WARNING: set WIFI_SSID, WIFI_PASS and DASH_HOST at the top of the sketch.");
}

static void dashboardSend(const Report& r) {
  if (reportQ && xQueueSend(reportQ, &r, 0) != pdTRUE) Serial.println("[dashboard] queue full - report dropped");
}
#endif

// ---------------- live loop ----------------
static uint32_t nextECGTime;
static long lastPredSample = 0;
static uint32_t maxProcUs = 0;
static int rawMin = 4095, rawMax = 0;          // diagnostics for the last 10 s
static long leadOffSamples = 0;
static long beatsAtLastPrint = 0;
#if DASHBOARD && ECG_STREAM
static uint16_t gEcg[ECG_MAX];                 // the last 10 s of raw samples, sent with the next report
static int      gEcgN = 0;
static uint8_t  gEcgDiv = 0;
#endif

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nFedApnea ESP32 - live sensor through the validated model");
  Serial.println("Predictions start after ~62 s of clean signal (60 s window + 2 s detector warm-up).");
  Serial.println("This is NOT a diagnosis: the model is ~0.65-0.68 balanced accuracy on held-out patients.");
  Serial.println("Every 10 s you get a 'signal' line (what the sensor sees) and a prediction line.");
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);
  notch50.init(50.0f, (float)FS, 10.0f);
  mlp_layout();
#if DASHBOARD
  dashboardStart();
#endif
  nextECGTime = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextECGTime) < 0) return;
  nextECGTime += ECG_PERIOD_US;

  // Electrodes off (lead-off): skip the sample but KEEP GOING to the 10 s report below, so the Pi and the
  // Serial Monitor are told "electrodes off" instead of the board going silent.
  if (digitalRead(LO_PLUS) || digitalRead(LO_MINUS)) {
    leadOffSamples++;
    pipelineLeadOff();
    notch50.reset();
  } else {
    int raw = analogRead(ECG_PIN);
#if DASHBOARD && ECG_STREAM
    if ((gEcgDiv++ % ECG_DIV) == 0 && gEcgN < ECG_MAX) gEcg[gEcgN++] = (uint16_t)raw;
#endif
    if (raw < rawMin) rawMin = raw;
    if (raw > rawMax) rawMax = raw;
    float f = notch50.process((float)raw);
    uint32_t t0 = micros();
    pipelineSample((double)f, (long)raw);
    uint32_t dt = micros() - t0;
    if (dt > maxProcUs) maxProcUs = dt;
  }

  if (gSample - lastPredSample >= (long)PRED_PERIOD_S * FS) {
    lastPredSample = gSample;
    long sec = gSample / FS;

    bool anyRaw = (rawMax >= rawMin);                // false if every sample in this 10 s was lead-off
    int  repSpan = anyRaw ? rawMax - rawMin : 0;     // kept for the dashboard report
    long repB10 = beatsTotal - beatsAtLastPrint;
    long repLo = leadOffSamples;

    // what the sensor and the detector saw in the last 10 s (a healthy resting heart: ~10-14 beats)
    Serial.printf("t=%lds signal: raw min/max %d/%d (span %d) | beats found in last 10 s: %ld | "
                  "lead-off samples: %ld | detector levels: signal %.3g, noise %.3g, threshold %.3g\n",
                  sec, anyRaw ? rawMin : 0, anyRaw ? rawMax : 0, repSpan, beatsTotal - beatsAtLastPrint, leadOffSamples,
                  det.spki, det.npki, det.threshold);
    beatsAtLastPrint = beatsTotal;
    rawMin = 4095; rawMax = 0; leadOffSamples = 0;

    float feat[8]; int n; float p; const char* why = ""; double cov;
    bool winOk = evaluateWindow(gSample, feat, &n, &p, &why, &cov);
    if (!winOk && repLo > 0) why = "lead-off: electrodes off or loose";
    if (winOk) {
      Serial.printf("t=%lds | %d beats, covering %.0f%% of the minute | mean_rr=%.3f sdnn=%.3f rmssd=%.3f pnn50=%.2f | "
                    "apnea probability=%.3f -> %s | slowest sample %lu us (budget 4000)\n",
                    sec, n, cov * 100.0, feat[0], feat[1], feat[2], feat[3], p,
                    p >= 0.5f ? "APNEA-LIKE" : "normal-like", (unsigned long)maxProcUs);
    } else {
      Serial.printf("t=%lds | no prediction: %s (valid beats: %d, covering %.0f%% of the minute)\n",
                    sec, why, n, cov * 100.0);
    }
#if DASHBOARD
    {
      static Report rep;             // static: keeps the large ECG buffer off the loop task's stack
      memset(&rep, 0, sizeof(rep));
      rep.tSec = sec; rep.valid = winOk ? 1 : 0; rep.nRR = n; rep.cov = (float)cov;
      if (winOk) { for (int j = 0; j < 8; j++) rep.feat[j] = feat[j]; rep.boardProb = p; }
      rep.span = repSpan; rep.beats10 = repB10; rep.leadoff = repLo;
      strncpy(rep.why, winOk ? "" : why, sizeof(rep.why) - 1);
#if ECG_STREAM
      rep.ecgN = gEcgN;
      memcpy(rep.ecg, gEcg, (size_t)gEcgN * sizeof(uint16_t));
      gEcgN = 0;
#endif
      dashboardSend(rep);
    }
#endif
    maxProcUs = 0;
  }
}
#else
// ---------------- REPLAY: samples arrive over USB as integer micro-millivolts, one per line ----------------
// No notch filter and no ADC here: the Python extraction that built the training data used neither.
#define CHUNK     500          // the laptop waits for an ACK after every 500 samples
#define PREROLL_S 10           // 10 s of signal before the first scored minute (detector warm-up)
static char lineBuf[24];
static int  lineLen = 0;
static long recvCount = 0;
static long winIdx = 0;

// A scored window ends at endSample once sample index endSample has been processed.
static void replayWindowCheck() {
  long s = gSample - 1;
  long first = (long)PREROLL_S * FS + (long)WINDOW_S * FS;
  if (s < first || (s - (long)PREROLL_S * FS) % ((long)WINDOW_S * FS) != 0) return;
  float feat[8]; int n; float p; const char* why = ""; double cov;
  bool ok = evaluateWindow(s, feat, &n, &p, &why, &cov);
  if (ok) {
    Serial.printf("RESULT %ld 1 %d %.12g", winIdx, n, cov);
    for (int j = 0; j < 8; j++) Serial.printf(" %.9g", feat[j]);
    Serial.printf(" %.9g\n", p);
  } else {
    Serial.printf("RESULT %ld 0 %d %.12g\n", winIdx, n, cov);
  }
  winIdx++;
}

void setup() {
  Serial.setRxBufferSize(8192);
  Serial.begin(115200);
  delay(1500);
  mlp_layout();
  pipelineResetAll();
  Serial.println("\nFedApnea ESP32 - REPLAY MODE (real PhysioNet ECG through the live pipeline)");
  Serial.println("READY-REPLAY");
}

void loop() {
  while (Serial.available()) {
    int c = Serial.read();
    if (c < 0) return;
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = 0;
      lineLen = 0;
      if (lineBuf[0] == 'R') {
        pipelineResetAll(); recvCount = 0; winIdx = 0;
        Serial.println("RESET_OK");
      } else if (lineBuf[0] == 'E') {
        Serial.println("REPLAY_DONE");
      } else if (lineBuf[0] != 0) {
        long q = atol(lineBuf);
        pipelineSample((double)q / 1e6, q);
        replayWindowCheck();
        recvCount++;
        if (recvCount % CHUNK == 0) Serial.printf("ACK %ld\n", recvCount);
      }
    } else if (lineLen < 22) {
      lineBuf[lineLen++] = (char)c;
    }
  }
}
#endif
