/*
  FedApnea ESP32 - does the board compute exactly what Python computes?
  ----------------------------------------------------------------------------
  No Wi-Fi, no sensors, no data to send. The exporter (export_model_for_esp32.py)
  puts two headers in this folder:
    model_export.h    trained weights + feature scaling constants (mean/std)
    validation_set.h  real PhysioNet test windows as RAW features, their true
                      labels, and the probability Python gave each window

  For every window the board scales the raw features with FEAT_MU/FEAT_SD (the
  same step a live sensor reading goes through), runs the model, and compares
  with Python:
    - the probability, window by window
    - the apnea/normal decision
    - each patient's balanced accuracy
  PASS means the board's numbers equal Python's on real patient data.

  Open the Serial Monitor at 115200. It re-runs every 15 s, so you can't miss it.
*/

#include <Arduino.h>
#include <math.h>
#include <stdint.h>

#if !__has_include("model_export.h") || !__has_include("validation_set.h")
#error "Run export_model_for_esp32.py first: it creates model_export.h and validation_set.h in this sketch folder."
#endif
#include "model_export.h"
#include "validation_set.h"

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

// Same forward pass as esp32_mlp_test / esp32_fl_node (matched Python to 5e-7).
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

static void validate() {
  int pos[VAL_N_PATIENTS] = {0}, neg[VAL_N_PATIENTS] = {0};
  int tp[VAL_N_PATIENTS] = {0}, tn[VAL_N_PATIENTS] = {0}, cnt[VAL_N_PATIENTS] = {0};
  double maxDiff = 0.0;
  int flips = 0, flipsNearThreshold = 0;

  unsigned long t0 = micros();
  for (int i = 0; i < VAL_N; i++) {
    float xn[8];
    for (int j = 0; j < 8; j++) xn[j] = (VAL_X[i][j] - FEAT_MU[j]) / FEAT_SD[j];
    float p = mlp_forward(MODEL_W, xn);

    double d = fabs((double)p - (double)VAL_P_PY[i]);
    if (d > maxDiff) maxDiff = d;

    int espDec = p >= 0.5f ? 1 : 0;
    int pyDec  = VAL_P_PY[i] >= 0.5f ? 1 : 0;
    if (espDec != pyDec) {
      flips++;
      if (fabsf(VAL_P_PY[i] - 0.5f) < 1e-4f) flipsNearThreshold++;
    }

    int k = VAL_PID[i];
    cnt[k]++;
    if (VAL_Y[i]) { pos[k]++; if (espDec == 1) tp[k]++; }
    else          { neg[k]++; if (espDec == 0) tn[k]++; }
  }
  unsigned long dtUs = micros() - t0;

  Serial.println("\n=== ESP32 vs Python on real PhysioNet test windows ===");
  Serial.println("patient  windows  python_bal_acc  esp32_bal_acc  difference");
  double sumEsp = 0.0;
  for (int k = 0; k < VAL_N_PATIENTS; k++) {
    double tpr = pos[k] ? (double)tp[k] / pos[k] : 0.0;
    double tnr = neg[k] ? (double)tn[k] / neg[k] : 0.0;
    double bal = 0.5 * (tpr + tnr);
    sumEsp += bal;
    Serial.printf("%-7s  %7d  %14.4f  %13.4f  %10.4f\n",
                  VAL_PATIENT_NAMES[k], cnt[k], (double)VAL_PY_BALACC[k], bal,
                  bal - (double)VAL_PY_BALACC[k]);
  }
  double meanEsp = sumEsp / VAL_N_PATIENTS;
  Serial.printf("mean balanced accuracy: python %.4f | esp32 %.4f\n",
                (double)VAL_PY_MEAN_BALACC, meanEsp);
  Serial.printf("largest probability difference over %d windows: %.2e\n", VAL_N, maxDiff);
  Serial.printf("decision flips: %d (%d of them within 1e-4 of the 0.5 threshold)\n",
                flips, flipsNearThreshold);
  Serial.printf("time per window (scale + model, includes bookkeeping): %.1f us\n",
                (double)dtUs / VAL_N);

  bool ok = (maxDiff < 1e-4) && (flips == flipsNearThreshold);
  if (ok && flips == 0) Serial.println("accuracy is identical to Python on every patient");
  Serial.println(ok ? "RESULT: PASS" : "RESULT: FAIL - the board does not reproduce Python's numbers");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nFedApnea ESP32 - model validation");
  mlp_layout();
}

void loop() {
  validate();
  delay(15000);
}
