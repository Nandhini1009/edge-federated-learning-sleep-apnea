/*
  FedApnea ESP32 node - milestone 9: REAL PhysioNet patient data on hardware
  ------------------------------------------------------------------------------
  Receives ONE real patient's HRV features + labels (exported by
  export_patient_for_esp32.py from apnea_ecg_features.csv) over Serial, then
  runs the SAME verified MLP training and the SAME verified networking as
  esp32_fl_node.ino - no synthetic patient, no placeholder labels. This is
  the hardware counterpart to fedavg_vs_fedper_real.py: the same patient,
  same split, same hyperparameters, so the resulting score is directly
  comparable to that script's per-client output for this record.

  HOW TO USE:
    1. python export_patient_for_esp32.py apnea_ecg_features.csv <record>
       (produces <record>_for_esp32.txt and prints a Python reference score)
    2. Upload this sketch, open Serial Monitor at 115200, wait for
       "READY - waiting for data..."
    3. Send the exported .txt file's lines to the board. The simplest way:
       Arduino IDE's Serial Monitor "Send" box can't send a whole file, so
       use a terminal that can (e.g. PuTTY's "Send file" over the matching
       COM port at 115200), OR run the companion send_patient_to_esp32.py
       (plain pyserial, no extra setup) which does this for you:
           python send_patient_to_esp32.py COM5 b05_for_esp32.txt
    4. Once it prints "received N_TRAIN + N_TEST rows, starting FL rounds",
       it behaves exactly like esp32_fl_node.ino from there: fetches the
       global model, trains, uploads, repeats.

  Serial Monitor: 115200 baud. Protocol: see PROTOCOL.md.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>
#include <string.h>

// ================= EDIT THESE =================
const char* WIFI_SSID   = "YOUR_WIFI_NAME";
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP   = "YOUR_SERVER_IP";
const int   SERVER_PORT = 8080;
const char* NODE_ID     = "esp32-replay-01";   // change per patient if running several in sequence
const int   ROUNDS_TO_RUN = 20;                 // matches fedavg_vs_fedper_real.py's ROUNDS
// ==============================================

const int   LOCAL_EPOCHS = 3;     // matches fedavg_vs_fedper_real.py / esp32_fl_node.ino
const float LEARNING_RATE = 0.05f;
const int   BATCH_SIZE = 32;

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
static float balanced_accuracy(const float* P, const float* X, const float* Y, int n) {
  float a1[16], a2[8];
  int tp = 0, tn = 0, pos = 0, neg = 0;
  for (int i = 0; i < n; i++) {
    int pred = mlp_forward(P, X + i * N_FEAT, a1, a2) >= 0.5f ? 1 : 0;
    if (Y[i] > 0.5f) { pos++; if (pred == 1) tp++; }
    else             { neg++; if (pred == 0) tn++; }
  }
  float tpr = pos ? (float)tp / pos : 0.0f;
  float tnr = neg ? (float)tn / neg : 0.0f;
  return 0.5f * (tpr + tnr);
}

// ---------------- REAL patient data, received over Serial ----------------
static float* FEATX_TR = nullptr;   // heap-allocated once sizes are known
static float* FEATY_TR = nullptr;
static float* FEATX_TE = nullptr;
static float* FEATY_TE = nullptr;
static int    N_TRAIN_ROWS = 0, N_TEST_ROWS = 0;
static float  POS_WEIGHT = 1.0f;
static int*   IDX = nullptr;

static bool parseFloatLine(const String& line, const char* tag, float* out, int nOut, int* labelOut) {
  if (!line.startsWith(tag)) return false;
  int pos = strlen(tag);
  for (int i = 0; i < nOut; i++) {
    while (pos < (int)line.length() && line[pos] == ' ') pos++;
    out[i] = line.substring(pos).toFloat();
    while (pos < (int)line.length() && line[pos] != ' ') pos++;
  }
  while (pos < (int)line.length() && line[pos] == ' ') pos++;
  *labelOut = line.substring(pos).toInt();
  return true;
}

// Reads the exported file's lines from Serial until "DONE". Allocates
// buffers once N_TRAIN/N_TEST are known (first two lines of the file).
static bool receivePatientData() {
  Serial.println("READY - waiting for data...");
  int trainFilled = 0, testFilled = 0;
  unsigned long lastByte = millis();
  while (true) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      lastByte = millis();
      if (line.length() == 0) continue;

      if (line.startsWith("N_TRAIN")) {
        N_TRAIN_ROWS = line.substring(8).toInt();
      } else if (line.startsWith("N_TEST")) {
        N_TEST_ROWS = line.substring(7).toInt();
      } else if (line.startsWith("POS_WEIGHT")) {
        POS_WEIGHT = line.substring(11).toFloat();
        // now that all three header values are in, allocate
        FEATX_TR = (float*)malloc(sizeof(float) * N_TRAIN_ROWS * N_FEAT);
        FEATY_TR = (float*)malloc(sizeof(float) * N_TRAIN_ROWS);
        FEATX_TE = (float*)malloc(sizeof(float) * N_TEST_ROWS * N_FEAT);
        FEATY_TE = (float*)malloc(sizeof(float) * N_TEST_ROWS);
        IDX = (int*)malloc(sizeof(int) * N_TRAIN_ROWS);
        if (!FEATX_TR || !FEATY_TR || !FEATX_TE || !FEATY_TE || !IDX) {
          Serial.printf("malloc failed for N_TRAIN=%d N_TEST=%d, free heap=%u\n",
                        N_TRAIN_ROWS, N_TEST_ROWS, (unsigned)ESP.getFreeHeap());
          return false;
        }
        Serial.printf("allocated for N_TRAIN=%d N_TEST=%d POS_WEIGHT=%.3f\n",
                      N_TRAIN_ROWS, N_TEST_ROWS, POS_WEIGHT);
      } else if (line.startsWith("TRAIN")) {
        float feats[N_FEAT]; int lab;
        if (parseFloatLine(line, "TRAIN", feats, N_FEAT, &lab) && trainFilled < N_TRAIN_ROWS) {
          memcpy(FEATX_TR + trainFilled * N_FEAT, feats, sizeof(feats));
          FEATY_TR[trainFilled] = (float)lab;
          trainFilled++;
        }
      } else if (line.startsWith("TEST")) {
        float feats[N_FEAT]; int lab;
        if (parseFloatLine(line, "TEST", feats, N_FEAT, &lab) && testFilled < N_TEST_ROWS) {
          memcpy(FEATX_TE + testFilled * N_FEAT, feats, sizeof(feats));
          FEATY_TE[testFilled] = (float)lab;
          testFilled++;
        }
      } else if (line.startsWith("DONE")) {
        Serial.printf("received %d/%d train rows, %d/%d test rows\n",
                      trainFilled, N_TRAIN_ROWS, testFilled, N_TEST_ROWS);
        return trainFilled == N_TRAIN_ROWS && testFilled == N_TEST_ROWS;
      }
    } else if (millis() - lastByte > 30000) {
      Serial.println("timed out waiting for data (30s of silence) - check the sender is running");
      return false;
    }
  }
}

// ---------------- networking (unchanged, verified against your aggregator) ----------------
static float   PARAMS[N_TOTAL];
static uint8_t rxbuf[10 + 4 * N_TOTAL];
static uint8_t txbuf[18 + 4 * N_TOTAL];

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

static void shuffleIdx(int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(lcg_uniform() * (float)(i + 1));
    if (j > i) j = i;
    int t = IDX[i]; IDX[i] = IDX[j]; IDX[j] = t;
  }
}

int roundsDone = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nFedApnea ESP32 node - REAL PhysioNet patient on hardware");

  mlp_layout();
  if (!receivePatientData()) {
    Serial.println("failed to receive patient data - halting. Reset and retry.");
    while (true) delay(1000);
  }
  for (int i = 0; i < N_TRAIN_ROWS; i++) IDX[i] = i;
  lcg_state = 12345u;
  mlp_init(PARAMS);

  connectWifi();
}

void loop() {
  if (roundsDone >= ROUNDS_TO_RUN) {
    delay(5000);
    return;
  }
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
    if (WiFi.status() != WL_CONNECTED) { delay(2000); return; }
  }

  unsigned long msR, msM, msU; int bytesM, bytesU;
  long r0 = getRound(msR);
  if (r0 < 0) { Serial.println("GET /round failed."); delay(2000); return; }

  uint32_t rnd = 0;
  int n = getModel(rnd, msM, bytesM);
  if (n < 0) { Serial.println("GET /model failed."); delay(2000); return; }

  unsigned long t0 = millis();
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    shuffleIdx(N_TRAIN_ROWS);
    mlp_train_epoch(PARAMS, FEATX_TR, FEATY_TR, IDX, N_TRAIN_ROWS, LEARNING_RATE, BATCH_SIZE, POS_WEIGHT);
  }
  unsigned long msT = millis() - t0;
  float score = balanced_accuracy(PARAMS, FEATX_TE, FEATY_TE, N_TEST_ROWS);

  int code = postUpdate(rnd, N_TRAIN_ROWS, score, n, msU, bytesU);
  Serial.printf("round %u | fetch %lu ms | train %lu ms | REAL PATIENT balanced_acc=%.3f | "
                "upload %d B -> HTTP %d %lu ms\n",
                (unsigned)rnd, msR + msM, msT, score, bytesU, code, msU);
  if (code != 200) { delay(1000); return; }

  roundsDone++;
  unsigned long tw = millis(); unsigned long dummy;
  while (millis() - tw < 90000) {
    long r = getRound(dummy);
    if (r > (long)rnd) break;
    delay(500);
  }

  if (roundsDone >= ROUNDS_TO_RUN) {
    Serial.printf("\n===== DONE: %d rounds on real patient data, final balanced_acc=%.3f =====\n",
                  roundsDone, score);
    Serial.println("Compare this against the Python per-client score for this same record");
    Serial.println("in fedavg_vs_fedper_real.py's output.");
  }
}
