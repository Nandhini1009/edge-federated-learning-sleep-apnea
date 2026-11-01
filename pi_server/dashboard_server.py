#!/usr/bin/env python3
"""
FedApnea decision server + live dashboard. Runs on the Raspberry Pi.

    python3 dashboard_server.py --model model_export.h --port 8081
    then open  http://<pi-address>:8081  on a laptop or phone on the same Wi-Fi.

What it does
  1. The ESP32 POSTs a small report to /ingest every 10 s: the 8 heart-rate-variability
     numbers of the last minute, signal-quality values, and what it is currently showing.
     By default the ECG waveform stays on the board (a 10 s preview is sent only if ECG_STREAM is set to 1).
  2. The Pi runs the model (the weights and scaling in model_export.h - the model your
     federated training produced) and takes the DECISION.
  3. Alert rule: ALERT when the model says "apnea-like" for --sustain windows in a row;
     it clears after --clear-after normal-like windows in a row. A single window above the
     threshold is only WATCH, because this model is weak (about 0.64 balanced accuracy).
  4. The decision goes back to the ESP32 in the reply; the board prints it and confirms in its next
     report what it received.
  5. The dashboard shows live values, the Pi's decision, what the ESP32 reports it is
     showing, an alert log, and has a button that sends a TEST alert to the ESP32.
  Every report is also appended to dashboard_log.csv.
  6. Optional cloud: the Pi can also push every decision to ThingSpeak (charts in a browser, alert email)
     and/or Blynk (phone dashboard, push alert). See the --thingspeak-* and --blynk-* options. The pushes
     run in a background thread, so a slow or broken cloud never delays the ESP32's reply.

Needs only numpy and the Python standard library. This is a research prototype for a
LAN demo: no login, no encryption, not a medical device.
"""
import argparse
import csv
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

STATE_NAMES = {0: "OK", 1: "WATCH", 2: "ALERT"}


def parse_header(path):
    s = open(path).read()

    def arr(name):
        m = re.search(rf"static const float {name}\[\d+\] = \{{(.*?)\}};", s, re.S)
        if not m:
            raise SystemExit(f"{path}: cannot find {name} - is this the model_export.h made by export_model_for_esp32.py?")
        return np.array([float(t.strip().rstrip("f")) for t in m.group(1).replace("\n", " ").split(",")
                         if t.strip()], dtype=np.float32)
    return arr("MODEL_W"), arr("FEAT_MU"), arr("FEAT_SD")


class Model:
    """The same 8-16-8-1 network the board runs, in float32."""

    def __init__(self, path):
        w, self.mu, self.sd = parse_header(path)
        if len(w) != 289 or len(self.mu) != 8 or len(self.sd) != 8:
            raise SystemExit("unexpected model size - expected 289 weights and 8 + 8 scaling values")
        self.w = w
        self.W1, self.b1 = w[0:128].reshape(16, 8), w[128:144]
        self.W2, self.b2 = w[144:272].reshape(8, 16), w[272:280]
        self.W3, self.b3 = w[280:288].reshape(1, 8), w[288:289]
        # a fingerprint of weights + scaling; the ESP32 sends its own so a mismatch is visible
        self.crc = "%08x" % (zlib.crc32(np.concatenate([w, self.mu, self.sd]).astype("<f4").tobytes()) & 0xFFFFFFFF)

    def prob(self, feat):
        x = ((np.asarray(feat, dtype=np.float32) - self.mu) / self.sd).astype(np.float32)
        a1 = np.maximum(self.W1 @ x + self.b1, 0).astype(np.float32)
        a2 = np.maximum(self.W2 @ a1 + self.b2, 0).astype(np.float32)
        z = (self.W3 @ a2 + self.b3).astype(np.float32)
        return float(1.0 / (1.0 + np.exp(-z[0])))


class Cloud:
    """Pushes the Pi's decisions to ThingSpeak and/or Blynk from a background thread.

    ThingSpeak (free plan): one update per channel every 15 s, 8 fields. We always send the LATEST
    report, never a backlog, and wait out the limit. An ALERT also sends an email through ThingSpeak's
    alerts API. Blynk: one batch update of 5 datastreams; an ALERT also logs the event 'apnea_alert',
    which Blynk turns into a push notification or email if you enabled that on the event.
    Keys are never printed or put on the dashboard.
    """

    def __init__(self, a):
        self.ts_key, self.ts_alert_key = a.thingspeak_key, a.thingspeak_alerts_key
        self.ts_url = a.thingspeak_url.rstrip("/")
        self.ts_interval = max(a.thingspeak_interval, 15.5)       # never faster than the free-plan limit
        self.bl_token = a.blynk_token
        self.bl_url = a.blynk_host.rstrip("/") if a.blynk_host.startswith("http") else "https://" + a.blynk_host
        self.bl_event, self.bl_interval = a.blynk_event, a.blynk_interval
        self.cooldown = a.alert_cooldown
        self.lock = threading.Lock()
        self.latest, self.seq, self.events = None, 0, []
        self.svc = {}
        if self.ts_key:
            self.svc["ThingSpeak"] = dict(ok=None, last_ok=None, error=None, sent=0)
        if self.bl_token:
            self.svc["Blynk"] = dict(ok=None, last_ok=None, error=None, sent=0)
        self.enabled = bool(self.svc)
        self.t_sent = {"ThingSpeak": 0.0, "Blynk": 0.0}
        self.seq_sent = {"ThingSpeak": 0, "Blynk": 0}
        self.level_sent = {"ThingSpeak": None, "Blynk": None}
        self.t_alert = {}                      # (service, kind) -> time of the last message of that kind
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    # ---- called from the request threads: never blocks
    def update(self, rec):
        if self.enabled:
            with self.lock:
                self.latest, self.seq = rec, self.seq + 1

    def notify(self, kind, text, force=False):
        if self.enabled:
            with self.lock:
                self.events.append((kind, text, force))

    def status(self):
        with self.lock:
            now = time.time()
            return [dict(name=n, ok=v["ok"], error=v["error"], sent=v["sent"],
                         last_ok_ago=None if v["last_ok"] is None else now - v["last_ok"])
                    for n, v in self.svc.items()]

    # ---- background thread
    def _run(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                latest, seq = self.latest, self.seq
                events, self.events = self.events, []
            now = time.time()
            for kind, text, force in events:
                if self.ts_key and self.ts_alert_key:
                    self._ts_alert(kind, text, now, force)
                if self.bl_token:
                    self._bl_alert(kind, text, now, force)
            if latest is None:
                continue
            level = latest.get("level")
            for name, interval, gap in (("ThingSpeak", self.ts_interval, 15.5), ("Blynk", self.bl_interval, 3.0)):
                if name not in self.svc or seq == self.seq_sent[name]:
                    continue
                since = now - self.t_sent[name]
                changed = self.level_sent[name] is not None and level != self.level_sent[name]
                if since >= interval or (changed and since >= gap):
                    ok = self._ts_update(latest) if name == "ThingSpeak" else self._bl_update(latest)
                    self.t_sent[name], self.seq_sent[name], self.level_sent[name] = time.time(), seq, level
                    self._result(name, ok)

    def _result(self, name, res):
        ok, err = res
        v = self.svc[name]
        with self.lock:
            changed = v["ok"] != ok
            v["ok"], v["error"] = ok, None if ok else err
            if ok:
                v["last_ok"], v["sent"] = time.time(), v["sent"] + 1
        if changed:
            print(f"[cloud] {name}: {'connected' if ok else 'PROBLEM - ' + str(err)}", flush=True)

    @staticmethod
    def _fetch(req, what):
        try:
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status, r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            return e.code, body
        except Exception as e:                                      # no internet, DNS, timeout ...
            return None, f"cannot reach the server ({type(e).__name__})"

    # ---- ThingSpeak
    def _ts_update(self, r):
        valid = r["valid"]
        f = r["f"] or []
        d = {"api_key": self.ts_key, "status": ("TEST " if r.get("test") else "") + r["state"],
             "field2": r["level"] if valid else -1}
        if valid:
            d.update(field1=round(r["p"], 4), field3=round(r["hr"] or 0, 1), field4=round(f[1], 4),
                     field5=round(f[2], 4), field6=round(f[3] * 100, 1))
        if r.get("cov") is not None:
            d["field7"] = round(r["cov"] * 100, 1)
        d["field8"] = r["ack"] if isinstance(r.get("ack"), int) else -1
        req = urllib.request.Request(self.ts_url + "/update.json", data=urllib.parse.urlencode(d).encode(), method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        code, body = self._fetch(req, "ThingSpeak")
        if code is None:
            return False, body
        try:
            entry = json.loads(body).get("entry_id", 0) if body.strip().startswith("{") else int(body.strip() or 0)
        except (ValueError, AttributeError):
            entry = 0
        if code == 200 and entry:
            return True, None
        return False, "ThingSpeak did not accept the update (wrong Write API key, or sent too fast)"

    def _ts_alert(self, kind, text, now, force):
        if not force and now - self.t_alert.get(("ThingSpeak", kind), 0.0) < self.cooldown:
            return
        self.t_alert[("ThingSpeak", kind)] = now
        subject = {"ALERT": "FedApnea ALERT: apnea-like pattern", "TEST": "FedApnea TEST alert",
                   "OFFLINE": "FedApnea: ESP32 offline"}.get(kind, "FedApnea notice")
        body = (text + " Research prototype - not a diagnosis.")[:500]
        req = urllib.request.Request(self.ts_url + "/alerts/send", data=json.dumps(dict(subject=subject[:60], body=body)).encode(),
                                     method="POST", headers={"ThingSpeak-Alerts-API-Key": self.ts_alert_key,
                                                             "Content-Type": "application/json"})
        code, resp = self._fetch(req, "ThingSpeak alert")
        if code is None or code >= 300:
            print(f"[cloud] ThingSpeak alert email NOT sent (HTTP {code})", flush=True)

    # ---- Blynk
    def _bl_update(self, r):
        valid = r["valid"]
        f = r["f"] or []
        pins = {"v1": r["level"] if valid else 0, "v4": ("TEST " if r.get("test") else "") + r["state"]}
        if valid:
            pins.update(v0=round(r["p"], 4), v2=int(round(r["hr"] or 0)), v3=round(f[1], 4))
        q = urllib.parse.urlencode({"token": self.bl_token, **pins}, quote_via=urllib.parse.quote)
        code, body = self._fetch(urllib.request.Request(f"{self.bl_url}/external/api/batch/update?{q}"), "Blynk")
        if code is None:
            return False, body
        if code == 200:
            return True, None
        try:
            msg = json.loads(body)["error"]["message"]
        except Exception:
            msg = f"HTTP {code}"
        return False, f"Blynk refused the update: {msg} (check the token, the server region, and that datastreams V0-V4 exist with the right types)"

    def _bl_alert(self, kind, text, now, force):
        if not force and now - self.t_alert.get(("Blynk", kind), 0.0) < self.cooldown:
            return
        self.t_alert[("Blynk", kind)] = now
        q = urllib.parse.urlencode({"token": self.bl_token, "code": self.bl_event, "description": f"{kind}: {text}"[:200]},
                                   quote_via=urllib.parse.quote)
        code, body = self._fetch(urllib.request.Request(f"{self.bl_url}/external/api/logEvent?{q}"), "Blynk event")
        if code != 200:
            print(f"[cloud] Blynk event NOT logged (HTTP {code}) - does the event '{self.bl_event}' exist in the template?", flush=True)


class Monitor:
    def __init__(self, model, threshold, sustain, clear_after, offline_after, log_path, cloud=None):
        self.cloud = cloud
        self.model, self.thr, self.sustain = model, threshold, sustain
        self.clear_after, self.offline_after = clear_after, offline_after
        self.lock = threading.Lock()
        self.hist = deque(maxlen=720)          # about 2 hours of 10 s reports
        self.alerts = deque(maxlen=100)
        self.level, self.streak, self.below = 0, 0, 0
        self.alert_id = 0
        self.last_seen = None
        self.offline_logged = False
        self.override_until, self.override_level = 0.0, 0
        self.last_reply = None                 # (time, level, state)
        self.last_ecg = None                   # latest raw-ECG preview from the board (display only, never logged)
        self.demo_until = 0.0                  # while a demo replay is running, the real board only listens
        self.device, self.device_crc, self.last_ack = None, None, None
        new_file = not os.path.exists(log_path)
        self.log_f = open(log_path, "a", newline="")
        self.log = csv.writer(self.log_f)
        if new_file:
            self.log.writerow(["time", "device", "valid", "why", "n_beats", "coverage", "hr_bpm"]
                              + [f"f{i}" for i in range(8)]
                              + ["board_prob", "pi_prob", "level", "state", "truth", "crc_match"])
        threading.Thread(target=self._watchdog, daemon=True).start()

    # ---- events
    def _event(self, kind, text, now):
        self.alerts.appendleft(dict(ts=now, kind=kind, text=text))

    def test_alert(self, seconds=30):
        with self.lock:
            now = time.time()
            self.override_until, self.override_level = now + seconds, 2
            self._event("TEST", f"test alert requested - the ESP32 gets it on its next report (up to 10 s) "
                                f"and prints ALERT for about {seconds} s", now)
        if self.cloud:
            self.cloud.notify("TEST", "This is a TEST alert from the FedApnea dashboard: the cloud alert path works.", force=True)

    def _check_offline(self, now):
        """Caller holds the lock. Returns True while the ESP32 is considered online."""
        online = self.last_seen is not None and now - self.last_seen <= self.offline_after
        if self.last_seen is not None and not online and not self.offline_logged:
            self.offline_logged = True
            self._event("OFFLINE", f"no report from the ESP32 for more than {self.offline_after:.0f} s", now)
            if self.cloud:
                self.cloud.update(dict(valid=False, state="OFFLINE", level=0, f=None, p=None, hr=None, cov=None, ack=None))
                self.cloud.notify("OFFLINE", f"No report from the ESP32 for more than {self.offline_after:.0f} s.")
        return online

    def _watchdog(self):                       # notices an offline ESP32 even when nobody has the dashboard open
        while True:
            time.sleep(2.0)
            with self.lock:
                self._check_offline(time.time())

    # ---- one report from the ESP32
    def ingest(self, rep):
        valid = bool(rep.get("valid"))
        p, feats = None, None
        if valid:
            feats = rep.get("f")
            if not isinstance(feats, list) or len(feats) != 8:
                raise ValueError("'f' must be a list of 8 numbers")
            feats = [float(v) for v in feats]
            if not all(math.isfinite(v) for v in feats):
                raise ValueError("features contain NaN/inf")
            p = self.model.prob(feats)

        is_demo = str(rep.get("device", "")).startswith("demo")
        with self.lock:
            now = time.time()
            if is_demo:
                self.demo_until = now + 25.0
            elif now < self.demo_until:
                # a demo replay is running: the real board still gets the current decision (so the alert
                # reaches it) and its confirmation is kept, but its own windows do not disturb the replay
                self.last_seen = now
                self.device_crc = rep.get("crc")
                ack = rep.get("ack")
                self.last_ack = ack if isinstance(ack, int) else None
                lvl = max(self.level, self.override_level if now < self.override_until else 0)
                self.last_reply = (now, lvl, STATE_NAMES[lvl])
                return dict(level=lvl, state=STATE_NAMES[lvl], p=-1, thr=self.thr)
            if self.offline_logged:
                self._event("INFO", "ESP32 is back online", now)
                self.offline_logged = False
            self.last_seen = now
            ecg = rep.get("ecg")
            if isinstance(ecg, list) and 0 < len(ecg) <= 4000 and all(isinstance(v, (int, float)) for v in ecg):
                self.last_ecg = dict(ts=now, hz=int(rep.get("ecg_hz", 125)), samples=[int(v) for v in ecg])
            self.device = str(rep.get("device", "esp32"))
            self.device_crc = rep.get("crc")
            ack = rep.get("ack")
            self.last_ack = ack if isinstance(ack, int) else None

            prev = self.level
            if valid:
                if p >= self.thr:
                    self.streak += 1; self.below = 0
                else:
                    self.below += 1; self.streak = 0
                if prev == 2:
                    self.level = 0 if self.below >= self.clear_after else 2
                else:
                    self.level = 2 if self.streak >= self.sustain else (1 if self.streak >= 1 else 0)
            else:
                self.streak = self.below = 0
                self.level = 0
            if prev < 2 and self.level == 2:
                self.alert_id += 1
                self._event("ALERT", f"sustained apnea-like pattern: probability {p:.2f} for {self.streak} windows in a row", now)
                raised = f"Sustained apnea-like pattern in the heart-rate variability: probability {p:.2f} for {self.streak} windows in a row."
                raised += f" Heart rate {60.0 / feats[0]:.0f} bpm." if feats and feats[0] > 0 else ""
            elif prev == 2 and self.level < 2:
                self._event("CLEARED", "back to normal-like" if valid else "signal lost", now)

            state = STATE_NAMES[self.level] if valid else ("WARMING UP" if str(rep.get("why", "")).startswith("less than") else "NO SIGNAL")
            reply_level, msg, test = self.level, state, False
            if now < self.override_until:
                reply_level, msg, test = max(self.level, self.override_level), "TEST ALERT", True
            self.last_reply = (now, reply_level, "TEST ALERT" if test else state)

            mean_rr = feats[0] if feats else 0.0
            rec = dict(ts=now, valid=valid, why=str(rep.get("why", "")), n=rep.get("n"), cov=rep.get("cov"),
                       hr=(60.0 / mean_rr) if mean_rr > 0 else None, f=feats, bp=rep.get("bp"), p=p,
                       level=self.level, state=state, span=rep.get("span"), b10=rep.get("b10"),
                       lo=rep.get("lo"), rssi=rep.get("rssi"), truth=rep.get("truth"), test=test, demo=is_demo, ack=self.last_ack,
                       crc_match=(self.device_crc == self.model.crc) if self.device_crc else None)
            self.hist.append(rec)
            if self.cloud:
                self.cloud.update(rec)
                if prev < 2 and self.level == 2:
                    self.cloud.notify("ALERT", raised)
            self.log.writerow([datetime.fromtimestamp(now).isoformat(timespec="seconds"), self.device,
                               int(valid), rec["why"], rec["n"], rec["cov"],
                               "" if rec["hr"] is None else f"{rec['hr']:.1f}"]
                              + (["%.6g" % v for v in feats] if feats else [""] * 8)
                              + [rec["bp"], "" if p is None else f"{p:.5f}", self.level, state,
                                 rec["truth"], rec["crc_match"]])
            self.log_f.flush()
            return dict(level=reply_level, state=msg, p=-1 if p is None else round(p, 4), thr=self.thr)

    # ---- what the dashboard asks for
    def snapshot(self):
        with self.lock:
            now = time.time()
            online = self._check_offline(now)
            last = self.hist[-1] if self.hist else None
            if self.last_seen is None:
                state = "WAITING"
            elif not online:
                state = "OFFLINE"
            else:
                state = last["state"]
            lr = self.last_reply
            return dict(
                now=now, state=state, online=online, level=self.level, alert_id=self.alert_id,
                device=self.device, last=last, seen_ago=None if self.last_seen is None else now - self.last_seen,
                sent=None if lr is None else dict(ago=now - lr[0], level=lr[1], state=lr[2]),
                esp_shows=self.last_ack,
                ecg=(self.last_ecg if self.last_ecg and now - self.last_ecg["ts"] < 40 else None), cloud=self.cloud.status() if self.cloud else [],
                model_crc=self.model.crc, device_crc=self.device_crc,
                history=[dict(ts=h["ts"], p=h["p"], hr=h["hr"], level=h["level"], valid=h["valid"])
                         for h in list(self.hist)[-240:]],
                alerts=[dict(ts=a["ts"], kind=a["kind"], text=a["text"]) for a in list(self.alerts)[:30]],
                config=dict(threshold=self.thr, sustain=self.sustain, clear_after=self.clear_after))


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FedApnea live monitor</title>
<style>
:root{--paper:#f3f0e8;--panel:#fbfaf6;--ink:#1f2a2e;--dim:#667075;--rule:#cfc9ba;--ok:#1f7a4d;--watch:#b7791f;--alert:#c0392b;--grey:#7a7a72;--line:#1d4e6b}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 "Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.mono,td.v,.big{font-family:Consolas,"SF Mono",Menlo,monospace}
header{background:var(--ink);color:#f3f0e8;padding:12px 20px;display:flex;flex-wrap:wrap;gap:8px 24px;align-items:baseline;justify-content:space-between}
h1{font:600 19px Georgia,"Times New Roman",serif;margin:0}
h1 small{font:400 13px "Segoe UI",sans-serif;opacity:.7;margin-left:10px}
.status{font-size:13px;display:flex;flex-wrap:wrap;gap:4px 18px}
.status span::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;margin-right:6px;opacity:.9}
main{max-width:1000px;margin:0 auto;padding:18px 20px 30px}
#demo{display:none;border:1px solid var(--watch);background:#f7ecd3;padding:8px 12px;margin-bottom:14px;font-size:14px}
.two{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-bottom:16px}
section{background:var(--panel);border:1px solid var(--rule);padding:14px 16px}
h2{font:600 13px "Segoe UI",sans-serif;margin:0 0 8px;color:var(--dim);border-bottom:1px solid var(--rule);padding-bottom:6px}
.big{font-size:40px;font-weight:700;line-height:1.1;margin:6px 0 4px}
.sub{color:var(--dim);font-size:13px}
#decision.alerting{background:var(--alert);border-color:var(--alert);color:#fff}
#decision.alerting h2,#decision.alerting .sub{color:#ffe3df;border-color:#e58a80}
#decision.alerting #state{color:#fff!important}
table{width:100%;border-collapse:collapse;font-size:14px}
th{font-weight:400;text-align:left;color:var(--dim);padding:6px 0;border-top:1px solid var(--rule);width:36%}
td{padding:6px 0;border-top:1px solid var(--rule)}
tr:first-child th,tr:first-child td{border-top:0}
td.v{font-size:19px;font-weight:600;width:24%}td.u{color:var(--dim);font-size:12.5px}
button{background:#fff;color:var(--ink);border:1px solid var(--ink);padding:7px 12px;font:14px "Segoe UI",sans-serif;cursor:pointer;margin:2px 6px 2px 0}
button:hover{background:var(--ink);color:#fff}
canvas{width:100%;height:180px;display:block}
.k-ALERT{color:var(--alert);font-weight:600}.k-CLEARED{color:var(--ok)}.k-TEST{color:var(--watch)}.k-OFFLINE{color:var(--grey)}
#log td{font-size:13px;vertical-align:top}#log td:first-child{width:90px;color:var(--dim)}#log td:nth-child(2){width:80px}
.note{color:var(--dim);font-size:12.5px;margin-top:18px;border-top:1px solid var(--rule);padding-top:10px}
</style></head><body>
<header><h1>FedApnea<small>federated sleep-apnea monitor &middot; ESP32 node &rarr; Raspberry Pi</small></h1>
<div class="status"><span id="conn">waiting for the ESP32...</span><span id="model">model: -</span><span id="cloud" style="display:none"></span></div></header>
<main>
<div id="demo"></div>
<div class="two">
 <section id="decision"><h2>Decision made by the Raspberry Pi</h2><div class="big" id="state">-</div>
  <div class="sub" id="prob">-</div><div class="sub" id="rule"></div></section>
 <section><h2>Sent back to the ESP32</h2><div class="big" style="font-size:30px" id="sent">-</div>
  <div class="sub" id="ack">-</div>
  <div style="margin-top:10px"><button id="snd">Enable alert sound</button><button id="tst">Send test alert to the ESP32</button></div></section>
</div>
<div class="two">
 <section><h2>Heart-rate variability (last 60 s window)</h2><table>
  <tr><th>Heart rate</th><td class="v" id="hr">-</td><td class="u">beats per minute</td></tr>
  <tr><th>Mean beat gap</th><td class="v" id="rr">-</td><td class="u">seconds</td></tr>
  <tr><th>SDNN</th><td class="v" id="sdnn">-</td><td class="u">overall variability (s)</td></tr>
  <tr><th>RMSSD</th><td class="v" id="rmssd">-</td><td class="u">beat-to-beat change (s)</td></tr>
  <tr><th>pNN50</th><td class="v" id="pnn">-</td><td class="u">share of big changes</td></tr></table></section>
 <section><h2>Signal and link</h2><table>
  <tr><th>Beats in window</th><td class="v" id="nb">-</td><td class="u" id="cov">-</td></tr>
  <tr><th>Signal span</th><td class="v" id="span">-</td><td class="u" id="b10">-</td></tr>
  <tr><th>Wi-Fi strength</th><td class="v" id="rssi">-</td><td class="u">dBm</td></tr>
  <tr><th>Board vs Pi</th><td class="v" id="agree">-</td><td class="u" id="agreesub">same model?</td></tr></table></section>
</div>
<section style="margin-bottom:16px"><h2>Raw ECG from the sensor <span id="ecgmeta" style="font-weight:400"></span></h2>
 <canvas id="c3" width="1400" height="260" style="height:210px"></canvas>
 <div class="sub" id="ecgnote">waiting for the first ECG preview from the board...</div></section>
<div class="two">
 <section><h2>Apnea-like probability (dashed line = alert threshold)</h2><canvas id="c1" width="700" height="200"></canvas></section>
 <section><h2>Heart rate (bpm)</h2><canvas id="c2" width="700" height="200"></canvas></section>
</div>
<section><h2>Event log</h2><table id="log"><tr><td class="sub">nothing yet</td></tr></table></section>
<div class="note">Research prototype, not a medical device and not a diagnosis. The model uses only heart-beat timing and scores about 0.64 balanced accuracy on held-out patients, so WATCH or ALERT means "the pattern looks like apnea to this model", nothing more.</div>
</main>
<script>
const $ = id => document.getElementById(id);
const COL = {OK:'#1f7a4d', WATCH:'#b7791f', ALERT:'#c0392b', 'TEST ALERT':'#c0392b', 'NO SIGNAL':'#7a7a72', 'WARMING UP':'#7a7a72', OFFLINE:'#7a7a72', WAITING:'#7a7a72'};
let lastAlertId = -1, soundOn = false, ctx = null;
function num(x, d){ return (x===null||x===undefined||isNaN(x)) ? '-' : Number(x).toFixed(d); }
function ago(s){ if(s===null||s===undefined) return '-'; return s<90 ? Math.round(s)+' s ago' : Math.round(s/60)+' min ago'; }
function hhmmss(t){ return new Date(t*1000).toLocaleTimeString(); }
function beep(){ if(!soundOn) return; try{ for(let i=0;i<3;i++){ const o=ctx.createOscillator(), g=ctx.createGain(); o.frequency.value=880; o.connect(g); g.connect(ctx.destination); g.gain.value=.2; const t=ctx.currentTime+i*.45; o.start(t); o.stop(t+.25);} }catch(e){} }
$('snd').onclick = () => { ctx = ctx || new (window.AudioContext||window.webkitAudioContext)(); soundOn = true; $('snd').textContent='Alert sound on'; beep(); };
$('tst').onclick = () => fetch('/api/test_alert', {method:'POST'}).then(tick);
function chart(cv, pts, lo, hi, thr){
  const g = cv.getContext('2d'), W = cv.width, H = cv.height; g.clearRect(0,0,W,H);
  g.strokeStyle='#cfc9ba'; g.lineWidth=1; g.fillStyle='#667075'; g.font='12px sans-serif';
  for(let i=0;i<=4;i++){ const y=10+(H-24)*i/4; g.beginPath(); g.moveTo(34,y); g.lineTo(W-6,y); g.stroke(); g.fillText((hi-(hi-lo)*i/4).toFixed(hi<=1?2:0), 2, y+4); }
  const Y = v => 10+(H-24)*(1-(v-lo)/(hi-lo)), X = i => 34+(W-40)*(pts.length<2?0:i/(pts.length-1));
  if(thr!==undefined){ g.setLineDash([6,4]); g.strokeStyle='#b7791f'; g.beginPath(); g.moveTo(34,Y(thr)); g.lineTo(W-6,Y(thr)); g.stroke(); g.setLineDash([]); }
  g.strokeStyle='#1d4e6b'; g.lineWidth=2; g.beginPath(); let pen=false;
  pts.forEach((v,i)=>{ if(v===null||v===undefined){pen=false;return;} const x=X(i), y=Y(Math.max(lo,Math.min(hi,v))); if(!pen){g.moveTo(x,y);pen=true;} else g.lineTo(x,y); });
  g.stroke();
}
function ecgChart(cv, e){
  const g = cv.getContext('2d'), W = cv.width, H = cv.height; g.clearRect(0,0,W,H);
  const x0 = 8, x1 = W-8, y0 = 8, y1 = H-8;
  if(!e || !e.samples || e.samples.length < 10){ g.fillStyle='#667075'; g.font='20px sans-serif'; g.fillText('no ECG preview (electrodes off, or the board is in private mode)', 20, H/2); return; }
  const v = e.samples, n = v.length, dur = n / e.hz;
  g.lineWidth = 1;                                   // ECG-paper grid: small box 0.04 s, big box 0.2 s
  for(let t=0; t<=dur+1e-6; t+=0.2){ const x = x0+(x1-x0)*t/dur; g.strokeStyle = (Math.round(t/0.2)%5===0) ? '#e3a8a0' : '#f0d4cf'; g.beginPath(); g.moveTo(x,y0); g.lineTo(x,y1); g.stroke(); }
  let lo = Math.min(...v), hi = Math.max(...v); const pad = Math.max(10, (hi-lo)*0.1); lo -= pad; hi += pad;
  for(let i=0;i<=4;i++){ const y = y0+(y1-y0)*i/4; g.strokeStyle = '#f0d4cf'; g.beginPath(); g.moveTo(x0,y); g.lineTo(x1,y); g.stroke(); }
  g.strokeStyle = '#1f2a2e'; g.lineWidth = 2; g.beginPath();
  v.forEach((a,i)=>{ const x = x0+(x1-x0)*i/(n-1), y = y1-(y1-y0)*(a-lo)/(hi-lo); i ? g.lineTo(x,y) : g.moveTo(x,y); });
  g.stroke();
}
function render(s){
  const L = s.last, st = s.state, col = COL[st] || '#7a7a72';
  $('conn').textContent = s.online ? ('ESP32 online - last report '+ago(s.seen_ago)) : (st==='WAITING' ? 'waiting for the ESP32...' : 'ESP32 OFFLINE - last report '+ago(s.seen_ago));
  $('conn').style.color = s.online ? '#8fe0b0' : '#c9c6bb';
  const match = s.device_crc ? (s.device_crc===s.model_crc) : null;
  $('model').textContent = 'model '+s.model_crc + (match===null ? '' : (match ? ' - same on the ESP32' : ' - ESP32 HAS A DIFFERENT MODEL'));
  $('model').style.color = match===false ? '#ff9b8f' : '';
  const shown = (L && L.test && st!=='OFFLINE') ? 'TEST ALERT' : st;
  $('state').textContent = shown; $('state').style.color = COL[shown] || col;
  $('prob').textContent = (L && L.valid) ? ('probability '+num(L.p,2)+' (alert above '+num(s.config.threshold,2)+')') : (L ? ('no usable window: '+(L.why||'')) : 'no report yet');
  $('rule').textContent = 'ALERT after '+s.config.sustain+' windows in a row above the threshold; clears after '+s.config.clear_after+' below.';
  $('decision').classList.toggle('alerting', st==='ALERT' || shown==='TEST ALERT');
  const dm = $('demo'); if(L && L.demo){ dm.style.display='block'; dm.textContent='Demo replay: these windows are recorded from a real PhysioNet patient'+(L.truth===1?' (true label: apnea)':(L.truth===0?' (true label: normal)':''))+', sent in place of the live sensor.'; } else { dm.style.display='none'; }
  if(s.alert_id !== lastAlertId){ if(lastAlertId !== -1 && s.alert_id>0){ beep(); if(navigator.vibrate) navigator.vibrate([300,150,300]); } lastAlertId = s.alert_id; }
  if(s.sent){ $('sent').textContent = s.sent.state; $('sent').style.color = COL[s.sent.state]||''; }
  $('ack').textContent = (s.esp_shows===null||s.esp_shows===undefined) ? 'ESP32 has not confirmed yet' : ('At its last report the ESP32 said it was showing: '+['OK','WATCH','ALERT'][s.esp_shows]+' (it confirms on every report)');
  const f = (L && L.valid && L.f) ? L.f : null;
  $('hr').textContent = L ? num(L.hr,0) : '-';
  $('rr').textContent = f ? num(f[0],3) : '-'; $('sdnn').textContent = f ? num(f[1],3) : '-';
  $('rmssd').textContent = f ? num(f[2],3) : '-'; $('pnn').textContent = f ? num(f[3]*100,0)+'%' : '-';
  $('nb').textContent = L ? (L.n ?? '-') : '-'; $('cov').textContent = (L && L.cov!=null) ? 'covering '+Math.round(L.cov*100)+'% of the minute' : '';
  $('span').textContent = L ? (L.span ?? '-') : '-'; $('b10').textContent = (L && L.b10!=null) ? L.b10+' beats in last 10 s' : '';
  $('rssi').textContent = L ? (L.rssi ?? '-') : '-';
  if(L && L.bp!=null && L.p!=null){ const d=Math.abs(L.bp-L.p); $('agree').textContent = d<1e-3 ? 'match' : 'DIFFERENT'; $('agree').style.color = d<1e-3 ? '#1f7a4d' : '#c0392b'; $('agreesub').textContent = 'board '+num(L.bp,3)+' / Pi '+num(L.p,3); }
  else { $('agree').textContent='-'; $('agreesub').textContent='same model?'; }
  const cl = $('cloud');
  if(s.cloud && s.cloud.length){ cl.style.display=''; const bad = s.cloud.filter(c=>c.ok===false);
    cl.textContent = 'cloud: '+s.cloud.map(c=>c.name+(c.ok===true?' ok':(c.ok===false?' PROBLEM':' ...'))).join(', ');
    cl.style.color = bad.length ? '#ff9b8f' : '#8fe0b0'; cl.title = bad.map(c=>c.name+': '+c.error).join('\n'); } else { cl.style.display='none'; }
  ecgChart($('c3'), s.ecg);
  $('ecgmeta').textContent = s.ecg ? ('- heart rate '+(L ? num(L.hr,0) : '-')+' bpm, last 10 s') : '';
  $('ecgnote').textContent = s.ecg ? 'Each report carries the last 10 seconds, so this view refreshes every 10 s. Shown for display only; the Pi does not store it or learn from it.' : 'No ECG preview right now (replay mode, electrodes off, or the board is in private mode).';
  chart($('c1'), s.history.map(h=>h.p), 0, 1, s.config.threshold);
  chart($('c2'), s.history.map(h=>h.hr), 40, 140);
  $('log').innerHTML = s.alerts.length ? s.alerts.map(a=>'<tr><td>'+hhmmss(a.ts)+'</td><td class="k-'+a.kind+'">'+a.kind+'</td><td>'+a.text+'</td></tr>').join('') : '<tr><td class="sub">nothing yet</td></tr>';
}
async function tick(){ try{ const r = await fetch('/api/state'); render(await r.json()); }catch(e){ $('conn').textContent='cannot reach the dashboard server'; } }
setInterval(tick, 2000); tick();
</script></body></html>'''


def make_handler(mon):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FedApneaDashboard/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, separators=(",", ":")), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(200, mon.snapshot())
            elif path == "/healthz":
                self._send(200, "ok\n", "text/plain")
            else:
                self._send(404, "not found\n", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            if path == "/ingest":
                try:
                    self._json(200, mon.ingest(json.loads(body.decode() or "{}")))
                except (ValueError, TypeError, KeyError) as e:
                    self._json(400, dict(error=str(e)))
            elif path == "/api/test_alert":
                mon.test_alert()
                self._json(200, dict(ok=True))
            else:
                self._send(404, "not found\n", "text/plain")
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_export.h")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--threshold", type=float, default=0.7, help="probability above which a window counts as apnea-like")
    ap.add_argument("--sustain", type=int, default=6, help="windows in a row above the threshold for an ALERT")
    ap.add_argument("--clear-after", type=int, default=3, help="windows in a row below the threshold to clear an ALERT")
    ap.add_argument("--offline-after", type=float, default=35.0, help="seconds without a report before OFFLINE")
    ap.add_argument("--log", default="dashboard_log.csv")
    g = ap.add_argument_group("cloud (optional) - keys can also come from environment variables so they stay out of your shell history")
    g.add_argument("--thingspeak-key", default=os.environ.get("THINGSPEAK_WRITE_KEY"), help="channel Write API key")
    g.add_argument("--thingspeak-alerts-key", default=os.environ.get("THINGSPEAK_ALERTS_KEY"),
                   help="account Alerts API key (Account > My Profile); enables the alert email")
    g.add_argument("--thingspeak-interval", type=float, default=20.0, help="seconds between updates (never below 15.5)")
    g.add_argument("--thingspeak-url", default="https://api.thingspeak.com")
    g.add_argument("--blynk-token", default=os.environ.get("BLYNK_TOKEN"), help="Blynk device auth token")
    g.add_argument("--blynk-host", default="blynk.cloud", help="your Blynk region server, e.g. blr1.blynk.cloud")
    g.add_argument("--blynk-event", default="apnea_alert", help="event code defined in your Blynk template")
    g.add_argument("--blynk-interval", type=float, default=30.0, help="seconds between routine updates (changes go out at once)")
    g.add_argument("--alert-cooldown", type=float, default=120.0, help="minimum seconds between cloud alert messages")
    a = ap.parse_args()

    model = Model(a.model)
    cloud = Cloud(a)
    mon = Monitor(model, a.threshold, a.sustain, a.clear_after, a.offline_after, a.log, cloud)
    if cloud.enabled:
        print("cloud push on: " + ", ".join(cloud.svc) + (" (+ alert email)" if a.thingspeak_key and a.thingspeak_alerts_key else "")
              + (f" (+ event '{a.blynk_event}')" if a.blynk_token else ""))
    print(f"FedApnea dashboard | model {model.crc} from {a.model} | threshold {a.threshold} "
          f"sustain {a.sustain} clear-after {a.clear_after} | port {a.port}")
    print(f"open http://<this machine's address>:{a.port} in a browser; the ESP32 reports to /ingest")
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(mon)).serve_forever()


if __name__ == "__main__":
    main()
