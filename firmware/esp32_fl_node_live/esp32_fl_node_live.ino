/*
  FedApnea ESP32 node - milestone 8: REAL ECG feeding the federated loop
  --------------------------------------------------------------------------
  This replaces esp32_fl_node.ino's synthetic patient generator with your
  own real, live HRV features (the verified pipeline from
  esp32_ecg_hrv_live.ino: 50 Hz notch -> streaming R-peak detector ->
  artifact rejection -> 8 HRV features per rolling window).

  Every three unchanged, previously-verified pieces are reused byte-for-byte:
    - ECG acquisition + streaming detector + HRV + artifact filter
      (esp32_ecg_hrv_live.ino, confirmed working on your hardware)
    - the 8-16-8-1 MLP core (esp32_mlp_test.ino, matched Python to 5e-7)
    - the Wi-Fi/HTTP networking (esp32_fl_node.ino, confirmed working
      against your aggregator, including a real 2-node federated run)

  *** IMPORTANT - READ BEFORE INTERPRETING ANY SCORE THIS PRINTS ***
  There is no real apnea/normal label for your own live ECG - you have not
  undergone a sleep study. The label attached to each captured window below
  is a PLACEHOLDER (alternating 0/1 by window index), which exists ONLY so
  the already-verified supervised-training code can run end-to-end on real
  feature vectors. It carries no medical meaning. What this sketch DOES
  legitimately prove: real ECG -> real R-peaks -> real HRV features ->
  real on-device training -> real federated upload, all in one pipeline on
  your own hardware. Accuracy numbers here are not evidence of anything
  about apnea detection; real labels come later from a PhysioNet replay.

  Known simplifications versus the production design (documented, not
  hidden): each Wi-Fi round-trip briefly pauses the ECG sampling loop for
  its duration (typically under a second on your network); the feature
  window used for this bench test is shorter/more frequent than the 60 s /
  patient-night spec in spec.py, purely so you can see the FL loop fire
  within a few minutes on the bench instead of waiting hours.

  Serial Monitor: 115200 baud.  Protocol: see PROTOCOL.md.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include <string.h>

struct Hrv { float mean_rr, sdnn, rmssd, pnn50, cv_rr, rr_range, median_rr, mean_abs_diff; };

// ================= EDIT THESE =================
const char* WIFI_SSID   = "YOUR_WIFI_NAME";
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP   = "YOUR_SERVER_IP";
const int   SERVER_PORT = 8080;
const char* NODE_ID     = "esp32-ecg-01";
// ==============================================

// ---------------- ECG pins (same as esp32_ecg_hrv_live.ino) ----------------
#define ECG_PIN   34
#define LO_PLUS   32
#define LO_MINUS  33
#define FS            250
#define INTEG_WIN_S   0.04f
#define REFRACTORY_S  0.25f
#define RR_MIN_S      0.3f
#define RR_MAX_S      2.0f
#define HRV_WINDOW_S  60          // length of the rolling window each feature vector summarises
const uint32_t ECG_PERIOD_US = 1000000UL / FS;

// ---- bench-test cadence (see the big comment above: NOT the production spec) ----
#define FEATURE_PERIOD_MS   20000   // try to capture a new feature vector this often
#define MIN_RR_FOR_FEATURE  8       // need at least this many real RR in the window
#define MAX_WINDOWS         30      // feature buffer capacity
#define TRAIN_MIN_WINDOWS   6       // don't attempt FL until at least this many real windows exist
#define NEW_WINDOWS_PER_ROUND 2     // attempt one more FL round every time this many NEW windows arrive

// ---------------- 50 Hz notch (unchanged, verified) ----------------
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

// ---------------- streaming detector (unchanged, verified PASS) ----------------
struct StreamingDetector {
  int   win;
  float sqRing[16];
  int   ringPos;
  float acc;
  float prevSample;
  int   warmupTarget;
  float warmupMax, warmupSum;
  int   warmupCount;
  bool  warmedUp;
  float spki, npki, threshold;
  bool  havePrevInteg;
  float prevInteg;
  bool  prevIsRise;
  int   lastPeakIdx;
  int   refractory;
  int   sampleIndex;

  void begin(int fs) {
    win = max(1, (int)(INTEG_WIN_S * fs));
    refractory = (int)(REFRACTORY_S * fs);
    warmupTarget = 2 * fs;
    for (int i = 0; i < 16; i++) sqRing[i] = 0.0f;
    ringPos = 0; acc = 0.0f; prevSample = 0.0f;
    warmupMax = -1e18f; warmupSum = 0.0f; warmupCount = 0; warmedUp = false;
    spki = npki = threshold = 0.0f;
    havePrevInteg = false; prevInteg = 0.0f; prevIsRise = false;
    lastPeakIdx = -refractory;
    sampleIndex = 0;
  }

  bool step(float x, int* outPeakIdx) {
    int i = sampleIndex++;
    float diff = (i == 0) ? 0.0f : (x - prevSample);
    prevSample = x;
    float sq = diff * diff;
    acc += sq;
    if (i >= win) acc -= sqRing[ringPos];
    sqRing[ringPos] = sq;
    ringPos = (ringPos + 1) % win;
    float integ_i = acc / (float)win;

    bool found = false;
    if (!warmedUp) {
      if (integ_i > warmupMax) warmupMax = integ_i;
      warmupSum += integ_i;
      warmupCount++;
      if (warmupCount >= warmupTarget) {
        spki = warmupMax * 0.5f;
        npki = warmupSum / (float)warmupCount;
        threshold = npki + 0.25f * (spki - npki);
        warmedUp = true;
      }
    } else if (havePrevInteg && i >= 2) {
      if (prevIsRise && prevInteg >= integ_i) {
        int peakIdx = i - 1;
        float peakVal = prevInteg;
        if (peakVal > threshold && peakIdx - lastPeakIdx > refractory) {
          lastPeakIdx = peakIdx;
          spki = 0.125f * peakVal + 0.875f * spki;
          *outPeakIdx = peakIdx;
          found = true;
        } else {
          npki = 0.125f * peakVal + 0.875f * npki;
        }
        threshold = npki + 0.25f * (spki - npki);
      }
    }
    if (havePrevInteg) prevIsRise = warmedUp ? (integ_i > prevInteg) : false;
    prevInteg = integ_i;
    havePrevInteg = true;
    return found;
  }
};
static StreamingDetector det;

// ---------------- rolling RR buffer (unchanged) ----------------
#define RING_CAP 200
static float    ringRR[RING_CAP];
static uint32_t ringTs[RING_CAP];
static int      ringHead = 0;
static int      ringCount = 0;

static void pushRR(float rr, uint32_t ts) {
  ringRR[ringHead] = rr;
  ringTs[ringHead] = ts;
  ringHead = (ringHead + 1) % RING_CAP;
  if (ringCount < RING_CAP) ringCount++;
}

// ---------------- artifact rejection (unchanged, verified against your real capture) ----------------
#define ARTIFACT_HIST      7
#define ARTIFACT_MIN_HIST  3
#define ARTIFACT_FRAC      0.25f
static float artifactHist[ARTIFACT_HIST];
static int   artifactHistCount = 0;
static int   artifactHistPos = 0;

static float medianOfHist() {
  float tmp[ARTIFACT_HIST];
  memcpy(tmp, artifactHist, artifactHistCount * sizeof(float));
  for (int i = 1; i < artifactHistCount; i++) {
    float key = tmp[i]; int j = i - 1;
    while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
    tmp[j + 1] = key;
  }
  int n = artifactHistCount;
  return (n % 2 == 1) ? tmp[n / 2] : 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
}
static void pushArtifactHist(float rr) {
  artifactHist[artifactHistPos] = rr;
  artifactHistPos = (artifactHistPos + 1) % ARTIFACT_HIST;
  if (artifactHistCount < ARTIFACT_HIST) artifactHistCount++;
}
static bool isArtifact(float rrS) {
  if (artifactHistCount < ARTIFACT_MIN_HIST) return false;
  float med = medianOfHist();
  if (med <= 0.0f) return false;
  return fabsf(rrS - med) / med > ARTIFACT_FRAC;
}

// ---------------- HRV (unchanged) ----------------
static void sortFloatsSmall(float* a, int n) {
  for (int i = 1; i < n; i++) {
    float key = a[i]; int j = i - 1;
    while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
    a[j + 1] = key;
  }
}
static Hrv computeHrv(const float* rr, int n) {
  Hrv h = {0, 0, 0, 0, 0, 0, 0, 0};
  if (n == 0) return h;
  double sum = 0.0;
  float mn = rr[0], mx = rr[0];
  for (int i = 0; i < n; i++) { sum += rr[i]; if (rr[i] < mn) mn = rr[i]; if (rr[i] > mx) mx = rr[i]; }
  h.mean_rr = (float)(sum / n);
  double varSum = 0.0;
  for (int i = 0; i < n; i++) { double d = rr[i] - h.mean_rr; varSum += d * d; }
  h.sdnn = (float)sqrt(varSum / n);
  int nd = n - 1;
  if (nd > 0) {
    double sq = 0.0, absSum = 0.0; int over50 = 0;
    for (int i = 0; i < nd; i++) {
      float d = rr[i + 1] - rr[i];
      sq += (double)d * d;
      absSum += fabsf(d);
      if (fabsf(d) > 0.05f) over50++;
    }
    h.rmssd = (float)sqrt(sq / nd);
    h.pnn50 = (float)over50 / (float)nd;
    h.mean_abs_diff = (float)(absSum / nd);
  }
  h.cv_rr = h.mean_rr > 0.0f ? h.sdnn / h.mean_rr : 0.0f;
  h.rr_range = mx - mn;
  static float tmp[RING_CAP];
  memcpy(tmp, rr, n * sizeof(float));
  sortFloatsSmall(tmp, n);
  h.median_rr = (n % 2 == 1) ? tmp[n / 2] : 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
  return h;
}
static int gatherWindow(uint32_t nowMs, float* out) {
  uint32_t cutoff = (nowMs > (uint32_t)(HRV_WINDOW_S * 1000)) ? nowMs - (uint32_t)(HRV_WINDOW_S * 1000) : 0;
  int n = 0;
  int idx = (ringHead - 1 + RING_CAP) % RING_CAP;
  for (int k = 0; k < ringCount; k++) {
    if (ringTs[idx] < cutoff) break;
    if (n < RING_CAP) out[n++] = ringRR[idx];
    idx = (idx - 1 + RING_CAP) % RING_CAP;
  }
  return n;
}

// ================= model spec (must match spec.py) =================
#define N_FEAT   8
#define N_LAYERS 3
#define N_BASE   280
#define N_TOTAL  289
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

static uint32_t lcg_state;
static float lcg_uniform() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}
static void mlp_init(float* P) {
  for (int l = 0; l < N_LAYERS; l++) {
    float a = sqrtf(6.0f / (float)L_IN[l]);
    for (int k = 0; k < L_IN[l] * L_OUT[l]; k++) P[L_W_OFF[l] + k] = (lcg_uniform() * 2.0f - 1.0f) * a;
    for (int k = 0; k < L_OUT[l]; k++) P[L_B_OFF[l] + k] = 0.0f;
  }
}

// ---------------- MLP core (unchanged, matched Python to 5e-7) ----------------
static float mlp_forward(const float* P, const float* x, float* a1, float* a2) {
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
static float G[N_TOTAL];
static void mlp_train_epoch(float* P, const float* X, const float* Y, const int* idx, int n,
                            float lr, int batch, float pw) {
  float a1[16], a2[8], dz2[8], dz1[16];
  for (int s = 0; s < n; s += batch) {
    int B = (n - s < batch) ? (n - s) : batch;
    memset(G, 0, sizeof(G));
    for (int k = 0; k < B; k++) {
      int id = idx[s + k];
      const float* x = X + id * N_FEAT;
      float y = Y[id];
      float p = mlp_forward(P, x, a1, a2);
      float dz3 = p * (1.0f - y) - pw * y * (1.0f - p);
      for (int i = 0; i < 8; i++) G[L_W_OFF[2] + i] += dz3 * a2[i];
      G[L_B_OFF[2]] += dz3;
      for (int i = 0; i < 8; i++) dz2[i] = (a2[i] > 0.0f) ? dz3 * P[L_W_OFF[2] + i] : 0.0f;
      for (int o = 0; o < 8; o++) {
        for (int i = 0; i < 16; i++) G[L_W_OFF[1] + o * 16 + i] += dz2[o] * a1[i];
        G[L_B_OFF[1] + o] += dz2[o];
      }
      for (int i = 0; i < 16; i++) {
        float sum = 0.0f;
        for (int o = 0; o < 8; o++) sum += dz2[o] * P[L_W_OFF[1] + o * 16 + i];
        dz1[i] = (a1[i] > 0.0f) ? sum : 0.0f;
      }
      for (int o = 0; o < 16; o++) {
        for (int i = 0; i < 8; i++) G[L_W_OFF[0] + o * 8 + i] += dz1[o] * x[i];
        G[L_B_OFF[0] + o] += dz1[o];
      }
    }
    float scale = lr / (float)B;
    for (int j = 0; j < N_TOTAL; j++) P[j] -= scale * G[j];
  }
}
static float balanced_accuracy(const float* P, const float* X, const float* Y, int from, int to) {
  float a1[16], a2[8];
  int tp = 0, tn = 0, pos = 0, neg = 0;
  for (int i = from; i < to; i++) {
    int pred = mlp_forward(P, X + i * N_FEAT, a1, a2) >= 0.5f ? 1 : 0;
    if (Y[i] > 0.5f) { pos++; if (pred == 1) tp++; }
    else             { neg++; if (pred == 0) tn++; }
  }
  float tpr = pos ? (float)tp / pos : 0.0f;
  float tnr = neg ? (float)tn / neg : 0.0f;
  return 0.5f * (tpr + tnr);
}

// ---------------- REAL feature buffer (replaces make_patient()) ----------------
static float FEATX[MAX_WINDOWS * N_FEAT];
static float FEATY[MAX_WINDOWS];     // PLACEHOLDER labels - see the big comment at the top
static int   windowCount = 0;

// ---------------- networking (unchanged, verified against your aggregator) ----------------
static float   PARAMS[N_TOTAL];
static uint8_t rxbuf[10 + 4 * N_TOTAL];
static uint8_t txbuf[18 + 4 * N_TOTAL];
int lastHttpCode = 0;

String baseUrl() { return String("http://") + SERVER_IP + ":" + SERVER_PORT; }

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) { delay(250); Serial.print("."); }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\nConnected. ESP32 IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi FAILED (wrong name/password, or not a 2.4 GHz network)");
  }
}

long getRound(unsigned long &ms) {
  HTTPClient http;
  unsigned long t0 = millis();
  http.begin(baseUrl() + "/round");
  http.setTimeout(5000);
  int code = http.GET();
  lastHttpCode = code;
  long r = -1;
  if (code == 200) {
    String body = http.getString();
    int k = body.indexOf(':');
    if (k >= 0) r = body.substring(k + 1).toInt();
  }
  http.end();
  ms = millis() - t0;
  return r;
}

int getModel(uint32_t &outRound, unsigned long &ms, int &bytes) {
  HTTPClient http;
  unsigned long t0 = millis();
  http.begin(baseUrl() + "/model?node=" + NODE_ID);
  http.setTimeout(5000);
  int code = http.GET();
  if (code != 200) { http.end(); return -1; }
  int len = http.getSize();
  if (len < 10 || len > (int)sizeof(rxbuf)) { http.end(); return -1; }
  WiFiClient* s = http.getStreamPtr();
  int got = 0;
  unsigned long tw = millis();
  while (got < len && millis() - tw < 5000) {
    int n = s->readBytes(rxbuf + got, len - got);
    if (n > 0) { got += n; tw = millis(); }
  }
  http.end();
  ms = millis() - t0;
  bytes = got;
  if (got != len || memcmp(rxbuf, "FLM1", 4) != 0) return -1;
  uint32_t rnd; uint16_t n;
  memcpy(&rnd, rxbuf + 4, 4);
  memcpy(&n, rxbuf + 8, 2);
  if (n == 0 || n > N_TOTAL || len != 10 + 4 * n) return -1;
  memcpy(PARAMS, rxbuf + 10, 4 * n);
  outRound = rnd;
  return n;
}

int postUpdate(uint32_t round, uint32_t nSamples, float score, int nParams,
               unsigned long &ms, int &bytes) {
  uint16_t n = (uint16_t)nParams;
  memcpy(txbuf, "FLU1", 4);
  memcpy(txbuf + 4, &round, 4);
  memcpy(txbuf + 8, &nSamples, 4);
  memcpy(txbuf + 12, &score, 4);
  memcpy(txbuf + 16, &n, 2);
  memcpy(txbuf + 18, PARAMS, 4 * nParams);
  bytes = 18 + 4 * nParams;
  HTTPClient http;
  unsigned long t0 = millis();
  http.begin(baseUrl() + "/update?node=" + NODE_ID);
  http.addHeader("Content-Type", "application/octet-stream");
  http.setTimeout(5000);
  int code = http.POST(txbuf, bytes);
  http.end();
  ms = millis() - t0;
  return code;
}

// ---------------- FL round on the REAL windows captured so far ----------------
const int   LOCAL_EPOCHS = 3;
const float LEARNING_RATE = 0.05f;
static int  IDX[MAX_WINDOWS];

static void shuffleIdx(int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(lcg_uniform() * (float)(i + 1));
    if (j > i) j = i;
    int t = IDX[i]; IDX[i] = IDX[j]; IDX[j] = t;
  }
}

static void attemptFLRound() {
  int trainCount = (windowCount * 7) / 10;
  if (trainCount < 1) trainCount = 1;
  int testCount = windowCount - trainCount;
  if (testCount < 1) { trainCount = windowCount - 1; testCount = 1; }

  int nPos = 0;
  for (int i = 0; i < trainCount; i++) nPos += (int)FEATY[i];
  float pw = (float)(trainCount - nPos) / (float)(nPos > 0 ? nPos : 1);
  for (int i = 0; i < trainCount; i++) IDX[i] = i;

  unsigned long msR, msM, msU;
  int bytesM, bytesU;
  long r0 = getRound(msR);
  if (r0 < 0) { Serial.println("[REAL-DATA FL] GET /round failed - server unreachable?"); return; }

  uint32_t rnd = 0;
  int n = getModel(rnd, msM, bytesM);
  if (n < 0) { Serial.println("[REAL-DATA FL] GET /model failed (bad response)."); return; }

  unsigned long t0 = millis();
  int batch = trainCount < 8 ? trainCount : 8;
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    shuffleIdx(trainCount);
    mlp_train_epoch(PARAMS, FEATX, FEATY, IDX, trainCount, LEARNING_RATE, batch, pw);
  }
  unsigned long msT = millis() - t0;
  float score = balanced_accuracy(PARAMS, FEATX, FEATY, trainCount, windowCount);

  int code = postUpdate(rnd, trainCount, score, n, msU, bytesU);
  Serial.printf("[REAL-DATA FL] round %u | windows=%d (train=%d test=%d) | fetch %lu ms | "
                "train %lu ms | placeholder-label score %.3f | upload %d B -> HTTP %d %lu ms\n",
                (unsigned)rnd, windowCount, trainCount, testCount, msR + msM, msT, score,
                bytesU, code, msU);
}

// ---------------- ECG state ----------------
static uint32_t nextECGTime;
static uint32_t ecgSampleIndex = 0;
static bool     needsRecalib = true;
static uint32_t lastPeakTs = 0;
static bool     havePeak = false;
static uint32_t lastFeatureMs = 0;
static float    windowBuf[RING_CAP];
static int      lastRoundAttemptWindowCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nFedApnea ESP32 node - REAL ECG feeding the federated loop");
  Serial.println("*** placeholder labels only - see the header comment before trusting any score ***");

  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);
  notch50.init(50.0f, (float)FS, 10.0f);

  mlp_layout();
  lcg_state = 12345u;
  mlp_init(PARAMS);

  nextECGTime = micros();
  connectWifi();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextECGTime) < 0) return;
  nextECGTime += ECG_PERIOD_US;

  uint32_t tsMs = ecgSampleIndex * (1000UL / FS);
  ecgSampleIndex++;

  if (digitalRead(LO_PLUS) || digitalRead(LO_MINUS)) { needsRecalib = true; return; }

  int rawADC = analogRead(ECG_PIN);

  if (needsRecalib) {
    det.begin(FS);
    notch50.reset();
    havePeak = false;
    needsRecalib = false;
  }

  float notched = notch50.process((float)rawADC);

  int peakIdx;
  if (det.step(notched, &peakIdx)) {
    uint32_t peakTs = (uint32_t)peakIdx * (1000UL / FS);
    if (havePeak) {
      float rrS = (float)(peakTs - lastPeakTs) / 1000.0f;
      if (rrS >= RR_MIN_S && rrS <= RR_MAX_S) {
        if (isArtifact(rrS)) {
          Serial.printf("beat @ %lu ms | RR=%.3f s | REJECTED (artifact)\n", (unsigned long)peakTs, rrS);
        } else {
          pushRR(rrS, peakTs);
          pushArtifactHist(rrS);
        }
      }
    }
    lastPeakTs = peakTs;
    havePeak = true;
  }

  if (tsMs - lastFeatureMs >= FEATURE_PERIOD_MS) {
    lastFeatureMs = tsMs;
    int n = gatherWindow(tsMs, windowBuf);
    if (n >= MIN_RR_FOR_FEATURE) {
      if (windowCount < MAX_WINDOWS) {
        Hrv h = computeHrv(windowBuf, n);
        float* fx = FEATX + windowCount * N_FEAT;
        fx[0] = h.mean_rr; fx[1] = h.sdnn; fx[2] = h.rmssd; fx[3] = h.pnn50;
        fx[4] = h.cv_rr;   fx[5] = h.rr_range; fx[6] = h.median_rr; fx[7] = h.mean_abs_diff;
        FEATY[windowCount] = (float)(windowCount % 2);   // PLACEHOLDER label, see header
        windowCount++;
        Serial.printf("captured REAL feature window #%d/%d (n=%d RR) mean_rr=%.3f sdnn=%.3f rmssd=%.3f\n",
                      windowCount, MAX_WINDOWS, n, h.mean_rr, h.sdnn, h.rmssd);
      } else {
        Serial.println("feature buffer full (MAX_WINDOWS) - not capturing more this run");
      }
    } else {
      Serial.printf("only %d RR in the last %ds - waiting for more beats before capturing a feature window\n",
                    n, HRV_WINDOW_S);
    }

    if (windowCount >= TRAIN_MIN_WINDOWS &&
        windowCount - lastRoundAttemptWindowCount >= NEW_WINDOWS_PER_ROUND) {
      lastRoundAttemptWindowCount = windowCount;
      if (WiFi.status() != WL_CONNECTED) connectWifi();
      if (WiFi.status() == WL_CONNECTED) attemptFLRound();
      else Serial.println("[REAL-DATA FL] skipped this attempt - WiFi not connected");
    }
  }
}
