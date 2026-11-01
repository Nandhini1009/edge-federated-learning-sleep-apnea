/*
  FedApnea ESP32 node - milestone 5: real on-device training in the federated loop
  ---------------------------------------------------------------------------------
  Each round the board:
    1. fetches the global shared weights from the aggregator (HTTP)
    2. trains the 8-16-8-1 MLP locally on its own patient's windows (fixed train split)
    3. scores balanced accuracy on its held-out test windows (last 30 %)
    4. uploads ONLY the shared weights (FedPer: base; the head stays on the board)
    5. waits until the round closes, then repeats

  The patient here is SYNTHETIC (generated on the board, same recipe as sim_nodes.py),
  so scores only prove the plumbing. Real Apnea-ECG features replace it later.

  Serial Monitor: 115200 baud.  Protocol: see PROTOCOL.md.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include <string.h>

// ================= EDIT THESE =================
const char* WIFI_SSID   = "YOUR_WIFI_NAME";       // 2.4 GHz network
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP   = "YOUR_SERVER_IP";       // IPv4 of the machine running fl_server.py (ipconfig / ifconfig)
const int   SERVER_PORT = 8080;
const char* NODE_ID     = "esp32-01";
const uint32_t NODE_SEED = 501;                   // different seed = different synthetic patient
const int   ROUNDS_TO_RUN = 15;                   // stop after this many completed rounds
// ==============================================

// Training settings (same as sim_nodes.py)
const int   LOCAL_EPOCHS = 3;
const float LEARNING_RATE = 0.05f;
const int   BATCH_SIZE = 32;

// ---------------- model spec (must match spec.py) ----------------
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

// ---------------- deterministic random numbers ----------------
static uint32_t lcg_state;
static float lcg_uniform() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}
static float lcg_gauss() {
  float u1 = lcg_uniform();
  if (u1 < 1e-7f) u1 = 1e-7f;
  float u2 = lcg_uniform();
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void mlp_init(float* P) {
  for (int l = 0; l < N_LAYERS; l++) {
    float a = sqrtf(6.0f / (float)L_IN[l]);
    for (int k = 0; k < L_IN[l] * L_OUT[l]; k++) P[L_W_OFF[l] + k] = (lcg_uniform() * 2.0f - 1.0f) * a;
    for (int k = 0; k < L_OUT[l]; k++) P[L_B_OFF[l] + k] = 0.0f;
  }
}

// ---------------- MLP core (verified against Python by esp32_mlp_test) ----------------
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

// One pass over n samples in the order given by idx[], SGD mini-batches, weighted BCE.
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

// Balanced accuracy on samples [from, to) of X/Y.
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

// ---------------- synthetic patient (same recipe as sim_nodes.py) ----------------
#define N_WIN   480
#define N_TRAIN 336                       // first 70 %, time-ordered
static float WX[N_WIN * N_FEAT];
static float WY[N_WIN];
static int   IDX[N_TRAIN];
static const float DELTA[N_FEAT] = {-1.043655f, 0.926305f, -0.638189f, -0.189972f,
                                    -0.055226f, -0.543064f, -1.002583f, 0.475634f};

static bool make_patient(uint32_t seed) {
  for (int attempt = 0; attempt < 50; attempt++) {
    lcg_state = seed * 2654435761u + (uint32_t)attempt * 97u + 12345u;
    for (int w = 0; w < 4; w++) lcg_uniform();               // warm up

    float prevalence = 0.20f + 0.35f * lcg_uniform();
    float p10 = 0.08f;
    float p01 = prevalence * p10 / (1.0f - prevalence);
    int st = 0;
    static uint8_t state[N_WIN];
    state[0] = 0;
    for (int t = 1; t < N_WIN; t++) {
      float u = lcg_uniform();
      st = (st == 0) ? (u < p01 ? 1 : 0) : (u < p10 ? 0 : 1);
      state[t] = (uint8_t)st;
    }
    float shift[N_FEAT];
    for (int j = 0; j < N_FEAT; j++) shift[j] = 0.8f * lcg_gauss();
    float noise = 0.8f + 0.4f * lcg_uniform();
    int trPos = 0, tePos = 0;
    for (int t = 0; t < N_WIN; t++) {
      for (int j = 0; j < N_FEAT; j++)
        WX[t * N_FEAT + j] = shift[j] + (float)state[t] * DELTA[j] + noise * lcg_gauss();
      WY[t] = (float)state[t];
      if (t < N_TRAIN) trPos += state[t]; else tePos += state[t];
    }
    float trFrac = (float)trPos / N_TRAIN;
    float teFrac = (float)tePos / (N_WIN - N_TRAIN);
    if (trFrac > 0.1f && trFrac < 0.9f && teFrac > 0.1f && teFrac < 0.9f) return true;
  }
  return false;
}

// ---------------- networking (same as the tested Wi-Fi client) ----------------
const int MAX_PARAMS = N_TOTAL;
static float   PARAMS[N_TOTAL];
static uint8_t rxbuf[10 + 4 * N_TOTAL];
static uint8_t txbuf[18 + 4 * N_TOTAL];
int lastHttpCode = 0;

String baseUrl() {
  return String("http://") + SERVER_IP + ":" + SERVER_PORT;
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(250);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\nConnected. ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal (RSSI dBm): ");
    Serial.println(WiFi.RSSI());
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
  uint32_t rnd;
  uint16_t n;
  memcpy(&rnd, rxbuf + 4, 4);
  memcpy(&n, rxbuf + 8, 2);
  if (n == 0 || n > MAX_PARAMS || len != 10 + 4 * n) return -1;
  memcpy(PARAMS, rxbuf + 10, 4 * n);       // overwrite the shared part; the head (if FedPer) stays
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

// ---------------- main flow ----------------
int roundsDone = 0;
float pw = 1.0f;
unsigned long sumFetchMs = 0, sumTrainMs = 0, sumUpMs = 0;
int lastDownBytes = 0, lastUpBytes = 0;
float scores[64];
bool summaryPrinted = false;

void shuffleIdx() {
  for (int i = N_TRAIN - 1; i > 0; i--) {
    int j = (int)(lcg_uniform() * (float)(i + 1));
    if (j > i) j = i;
    int t = IDX[i]; IDX[i] = IDX[j]; IDX[j] = t;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nFedApnea ESP32 node - federated training");

  mlp_layout();
  if (!make_patient(NODE_SEED)) {
    Serial.println("could not build a usable synthetic patient; change NODE_SEED");
    while (true) delay(1000);
  }
  int nPos = 0;
  for (int i = 0; i < N_TRAIN; i++) nPos += (int)WY[i];
  pw = (float)(N_TRAIN - nPos) / (float)(nPos > 0 ? nPos : 1);
  for (int i = 0; i < N_TRAIN; i++) IDX[i] = i;

  lcg_state = NODE_SEED * 7919u + 3u;
  mlp_init(PARAMS);                       // this node's own head lives in PARAMS[280..288]
  Serial.printf("patient: %d train / %d test windows, train positives %d, pos_weight %.3f\n",
                N_TRAIN, N_WIN - N_TRAIN, nPos, pw);
  connectWifi();
}

void loop() {
  if (roundsDone >= ROUNDS_TO_RUN) {
    if (!summaryPrinted) {
      summaryPrinted = true;
      Serial.println("\n===== SUMMARY =====");
      Serial.printf("rounds completed      : %d\n", roundsDone);
      Serial.printf("avg fetch (round+model): %lu ms  (%d bytes down)\n", sumFetchMs / roundsDone, lastDownBytes);
      Serial.printf("avg local training     : %lu ms  (%d epochs x %d windows)\n", sumTrainMs / roundsDone, LOCAL_EPOCHS, N_TRAIN);
      Serial.printf("avg upload             : %lu ms  (%d bytes up)\n", sumUpMs / roundsDone, lastUpBytes);
      Serial.printf("free heap              : %u bytes\n", (unsigned)ESP.getFreeHeap());
      Serial.print("test balanced accuracy per round:");
      for (int i = 0; i < roundsDone && i < 64; i++) Serial.printf(" %.3f", scores[i]);
      Serial.println();
    }
    delay(1000);
    return;
  }

  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
    if (WiFi.status() != WL_CONNECTED) { delay(2000); return; }
  }

  unsigned long msR = 0, msM = 0, msU = 0;
  int bytesM = 0, bytesU = 0;

  long r0 = getRound(msR);
  if (r0 < 0) {
    Serial.print("GET /round failed. URL: ");
    Serial.print(baseUrl());
    Serial.print("/round  code: ");
    Serial.println(lastHttpCode);
    delay(2000);
    return;
  }

  uint32_t rnd = 0;
  int n = getModel(rnd, msM, bytesM);
  if (n < 0) {
    Serial.println("GET /model failed (bad response).");
    delay(2000);
    return;
  }

  // ---- local training on this node's patient ----
  unsigned long t0 = millis();
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    shuffleIdx();
    mlp_train_epoch(PARAMS, WX, WY, IDX, N_TRAIN, LEARNING_RATE, BATCH_SIZE, pw);
  }
  unsigned long msT = millis() - t0;
  float score = balanced_accuracy(PARAMS, WX, WY, N_TRAIN, N_WIN);

  int code = postUpdate(rnd, N_TRAIN, score, n, msU, bytesU);
  Serial.printf("round %u | fetch %lu ms | train %lu ms | score %.3f | upload %d B -> HTTP %d %lu ms\n",
                (unsigned)rnd, msR + msM, msT, score, bytesU, code, msU);
  if (code != 200) {
    delay(1000);
    return;
  }

  if (roundsDone < 64) scores[roundsDone] = score;
  roundsDone++;
  sumFetchMs += msR + msM;
  sumTrainMs += msT;
  sumUpMs += msU;
  lastDownBytes = bytesM;
  lastUpBytes = bytesU;

  // Wait until the server closes this round.
  unsigned long tw = millis();
  unsigned long dummy;
  while (millis() - tw < 90000) {
    long r = getRound(dummy);
    if (r > (long)rnd) break;
    delay(500);
  }
}
