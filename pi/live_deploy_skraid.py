#!/usr/bin/env python3
"""
SKRAID — Live Deployment: Real-Time Skid Risk Inference Loop (Raspberry Pi 4)
================================================================================
Runs the actual passive-warning L1 ADAS loop: live camera thread + live IMU
serial thread feed rolling buffers; a background vision-feature thread
continuously computes per-frame features ONCE as each frame arrives; and a
main loop fires a new risk score every STEP_SEC seconds, each one pooled
over the trailing WINDOW_SEC seconds of already-computed features.

⚠️ WINDOW_SEC vs STEP_SEC — READ BEFORE CHANGING EITHER:
WINDOW_SEC (2.0) is the span the model was TRAINED on — every feature
(percentile pooling, VIBRATION_ENERGY, JERK_MAX, etc.) assumes ~2 seconds
of data. This must stay 2.0 or predictions drift off the trained feature
distribution. STEP_SEC is just how often you re-score that same trailing
2s window — safe to shrink for faster updates. Windows now OVERLAP (a
0.5s step means ~75% of each window's frames were already in the last
one) — that's expected and fine because vision features are computed
once per frame, not once per window (see VisionFeatureThread below).

⚠️ IMU DERIVATION NOTE (unchanged from earlier version):
gx, gy, gz are DERIVED as diff(roll)/dt, diff(pitch)/dt, diff(yaw)/dt from
the YPR stream, matching Cell 9 exactly — not the Razor's raw gyro axes.

TWO MODES:
  --dry-run   : replays a recorded (video, fusion_csv) pair at real-world
                pace through the SAME threaded pipeline, in place of live
                camera/serial.
  (default)   : live camera (cv2.VideoCapture) + live IMU serial
                (/dev/ttyUSB0). GPS is NOT used — SPEED had ~zero variance
                in the trained data and isn't a model feature.

USAGE:
  Dry run (safe, no hardware), scoring every 0.5s:
    python3 live_deploy_skraid.py --dry-run \\
        --video session.mp4 --fusion-csv fusion_session.csv \\
        --model rf_model.joblib --step-sec 0.5

  Live on the scooter, scoring every 1s:
    python3 live_deploy_skraid.py \\
        --model rf_model.joblib --camera-index 0 --imu-port /dev/ttyUSB0 \\
        --step-sec 1.0

  Live with GPIO buzzer/LED wired:
    python3 live_deploy_skraid.py --model rf_model.joblib --gpio \\
        --buzzer-pin 18 --led-pin 23 --step-sec 0.5
"""

import argparse
import csv
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from skraid_ekf import RiskEKF, tree_disagreement

HIGH_RISK_THRESHOLD = 0.15
IMU_RATE_ASSUMED = 20.0
VISION_TOP_RATIO = 0.4


# ============================================================
# VISION FEATURE FUNCTIONS (per-frame; identical math to benchmark_skraid_pi4.py)
# ============================================================
def grey_world_wb(frame_bgr):
    result = frame_bgr.copy().astype(np.float32)
    mean_b, mean_g, mean_r = (np.mean(result[:, :, i]) for i in range(3))
    mean_grey = (mean_b + mean_g + mean_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (mean_grey / (mean_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (mean_grey / (mean_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (mean_grey / (mean_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def frame_road_roi(frame, top_ratio=VISION_TOP_RATIO, bot_ratio=1.0):
    h, w = frame.shape[:2]
    return frame[int(h * top_ratio):int(h * bot_ratio), :]


def compute_optical_flow(prev_gray, curr_gray):
    """Both inputs are already at the working resolution set by ROI_SCALE
    upstream in extract_frame_features -- no resize happens here anymore."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return {"flow_mag_mean": np.mean(mag), "flow_mag_std": np.std(mag)}


def compute_edge_features(gray):
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    return {"edge_density": edge_density, "edge_magnitude_mean": np.mean(edge_magnitude)}


def compute_hsv_features(frame_roi_wb):
    hsv = cv2.cvtColor(frame_roi_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return {"hue_mean": np.mean(h), "saturation_mean": np.mean(s), "value_mean": np.mean(v)}


def compute_wetness_index(frame_roi_wb, gray_roi):
    hsv = cv2.cvtColor(frame_roi_wb, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    wet_mask_tar = (s < 50) & (v > 120)
    wet_mask_mud = (h > 10) & (h < 40) & (s > 40) & (s < 220) & (v > 80) & (v < 210)
    wet_mask_water = (s < 30) & (v > 150)
    wetness_ratio = max(
        np.sum(wet_mask_tar) / wet_mask_tar.size,
        np.sum(wet_mask_mud) / wet_mask_mud.size,
        np.sum(wet_mask_water) / wet_mask_water.size,
    )
    bright_threshold = np.percentile(gray_roi, 95)
    specular_ratio = np.sum(gray_roi > bright_threshold) / gray_roi.size
    brightness_uniformity = 1.0 / (np.std(gray_roi) + 1)
    brightness_variance = float(np.var(gray_roi))
    mud_score = float(np.mean(wet_mask_mud.astype(np.float32)))
    return {
        "specular_ratio": specular_ratio,
        "wetness_ratio": wetness_ratio,
        "brightness_uniformity": brightness_uniformity,
        "brightness_variance": brightness_variance,
        "mud_score": mud_score,
    }


def compute_texture_features(gray_roi):
    k = 15
    mean_filter = cv2.blur(gray_roi, (k, k))
    variance = (gray_roi.astype(float) - mean_filter) ** 2
    local_std = np.sqrt(cv2.blur(variance, (k, k)))
    return {"texture_roughness": np.mean(local_std)}


def compute_color_features(frame_roi_wb):
    lab = cv2.cvtColor(frame_roi_wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return {"lightness_mean": np.mean(l), "color_variance": np.var(a) + np.var(b)}


def extract_frame_features(prev_gray, frame_bgr, top_ratio=VISION_TOP_RATIO, roi_scale=0.5):
    """ROI_SCALE FIX: resize once here, before ANY feature function, so flow,
    edge, texture, HSV, and color all run on the same downscaled image.
    (Previously only flow was downscaled; edge/texture stayed full-res and
    became the bottleneck once flow was cut -- measured 1.28x real Pi
    speedup instead of the expected 4x+ from downscaling flow alone.)"""
    frame_wb = grey_world_wb(frame_bgr)
    frame_roi = frame_road_roi(frame_wb, top_ratio)
    if roi_scale != 1.0:
        h, w = frame_roi.shape[:2]
        frame_roi = cv2.resize(frame_roi, (int(w * roi_scale), int(h * roi_scale)),
                                interpolation=cv2.INTER_AREA)
    gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    feats = {}
    feats.update(compute_optical_flow(prev_gray, gray_roi))
    feats.update(compute_edge_features(gray_roi))
    feats.update(compute_hsv_features(frame_roi))
    feats.update(compute_wetness_index(frame_roi, gray_roi))
    feats.update(compute_texture_features(gray_roi))
    feats.update(compute_color_features(frame_roi))
    return feats, gray_roi


def pool_vision_window(frame_feats_list):
    """Percentile pooling — see benchmark_skraid_pi4.py for rationale."""
    df = pd.DataFrame(frame_feats_list)
    row = {}
    row["EDGE_DENSITY_MEAN"] = np.percentile(df["edge_density"], 10)
    row["EDGE_MAGNITUDE_MEAN"] = df["edge_magnitude_mean"].mean()
    row["HUE_MEAN"] = df["hue_mean"].mean()
    row["SATURATION_MEAN"] = df["saturation_mean"].mean()
    row["VALUE_MEAN"] = df["value_mean"].mean()
    row["SPECULAR_RATIO"] = df["specular_ratio"].mean()
    row["BRIGHTNESS_UNIFORMITY"] = df["brightness_uniformity"].mean()
    row["TEXTURE_ROUGHNESS"] = np.percentile(df["texture_roughness"], 10)
    row["LIGHTNESS_MEAN"] = df["lightness_mean"].mean()
    row["COLOR_VARIANCE"] = df["color_variance"].mean()
    row["BRIGHTNESS_VARIANCE"] = df["brightness_variance"].mean()
    wetness_ratio_p90 = np.percentile(df["wetness_ratio"], 90)
    mud_score_p90 = np.percentile(df["mud_score"], 90)
    row["MUD_SCORE"] = mud_score_p90
    flow_var = df["flow_mag_std"].mean() ** 2
    row["ROUGHNESS_INDEX"] = flow_var * row["EDGE_DENSITY_MEAN"]
    row["WET_INDICATOR"] = wetness_ratio_p90 * row["SPECULAR_RATIO"]
    return row


def extract_imu_window(ax, ay, az, roll, pitch, yaw, dt):
    """gx/gy/gz derived from YPR diffs — matches Cell 9. See module docstring."""
    accel_mag = np.sqrt(np.asarray(ax) ** 2 + np.asarray(ay) ** 2 + np.asarray(az) ** 2)
    roll_diff = np.diff(roll)
    pitch_diff = np.diff(pitch)
    yaw_diff = np.diff(yaw)
    gx = np.pad(roll_diff / dt, (0, len(ax) - len(roll_diff)), "edge")
    gy = np.pad(pitch_diff / dt, (0, len(ay) - len(pitch_diff)), "edge")
    gz = np.pad(yaw_diff / dt, (0, len(az) - len(yaw_diff)), "edge")
    gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return {
        "ACCEL_MEAN": np.mean(accel_mag),
        "ACCEL_STD": np.std(accel_mag),
        "ACCEL_MAX": np.max(accel_mag),
        "ACCEL_Z_VAR": np.var(az),
        "VIBRATION_ENERGY": np.sum(np.diff(accel_mag) ** 2),
        "JERK_MEAN": np.mean(np.abs(np.diff(accel_mag) / dt)),
        "JERK_MAX": np.max(np.abs(np.diff(accel_mag) / dt)),
        "GYRO_MEAN": np.mean(gyro_mag),
        "GYRO_STD": np.std(gyro_mag),
        "GYRO_MAX": np.max(gyro_mag),
        "ROLL_RATE_MEAN": np.mean(np.abs(gx)),
        "ROLL_RATE_STD": np.std(gx),
        "PITCH_RATE_MEAN": np.mean(np.abs(gy)),
        "PITCH_RATE_STD": np.std(gy),
        "YAW_RATE_MEAN": np.mean(np.abs(gz)),
        "STABILITY_SCORE": np.std(accel_mag) + np.std(gyro_mag),
        "ROUGHNESS_INDEX_IMU": np.sum(np.diff(accel_mag) ** 2) * np.var(gyro_mag),
    }


# ============================================================
# WARNING CUE — console+CSV always fire; GPIO pulses only on rising edge
# (prevents buzzer chattering every STEP_SEC during one multi-step hazard),
# GPIO best-effort, never blocks/crashes the loop.
# ============================================================
class WarningCue:
    def __init__(self, use_gpio, buzzer_pin, led_pin, log_path):
        self.use_gpio = use_gpio
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.gpio_ready = False
        self.GPIO = None

        if self.use_gpio:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.buzzer_pin, GPIO.OUT)
                GPIO.setup(self.led_pin, GPIO.OUT)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                GPIO.output(self.led_pin, GPIO.LOW)
                self.gpio_ready = True
                print(f"[GPIO] Ready — buzzer=BCM{buzzer_pin}, led=BCM{led_pin}")
            except Exception as e:
                print(f"[GPIO] Unavailable ({e}) — falling back to console+CSV only")
                # On Pi 5, the classic RPi.GPIO package targets the Pi 4's GPIO
                # chip and does not reliably work against the Pi 5's RP1
                # controller -- this typically shows up as an import error or
                # a runtime failure right here, not a crash elsewhere. Fix:
                #   pip uninstall RPi.GPIO
                #   pip install rpi-lgpio
                # rpi-lgpio installs itself under the SAME "RPi.GPIO" import
                # name (drop-in compatible), so no code change is needed --
                # only the installed package differs between Pi 4 and Pi 5.
                print("[GPIO] If this is a Raspberry Pi 5: the classic RPi.GPIO "
                      "package often fails against the Pi 5's RP1 GPIO chip. Try:")
                print("       pip uninstall RPi.GPIO && pip install rpi-lgpio")
                print("       (rpi-lgpio is a drop-in replacement -- no code change needed)")
                self.gpio_ready = False

        self.log_path = log_path
        self._init_log()

    def _init_log(self):
        new_file = not Path(self.log_path).exists()
        self._log_f = open(self.log_path, "a", newline="")
        self._writer = csv.writer(self._log_f)
        if new_file:
            self._writer.writerow(
                ["timestamp", "step_id", "proba_high_risk", "flagged_high_risk",
                 "predicted_class", "vision_pool_ms", "imu_ms",
                 "inference_ms", "ekf_ms", "total_ms",
                 "ekf_smooth", "ekf_alarm", "ekf_rate"]
            )

    def fire(self, step_id, proba_high_risk, flagged, predicted_class,
             vision_ms, imu_ms, inference_ms, total_ms,
             ekf_ms=0.0, ekf_smooth=None, ekf_alarm=None, ekf_rate=None):
        ts = datetime.now().isoformat(timespec="milliseconds")

        # GPIO/visible alarm is gated on the EKF's hysteresis-debounced decision
        # when available -- the smoothed, hold-timed signal is what should reach
        # the rider, not every raw per-window flag (that chatters).
        gpio_trigger = ekf_alarm if ekf_alarm is not None else flagged

        tag = "⚠️  HIGH RISK" if gpio_trigger else "   ok        "
        ekf_str = f" | ekf={ekf_smooth:.3f} alarm={ekf_alarm}" if ekf_smooth is not None else ""
        print(f"[{ts}] step={step_id:5d} {tag} "
              f"p(HR)={proba_high_risk:.3f} class={predicted_class:9s} "
              f"| vision_pool={vision_ms:5.2f}ms imu={imu_ms:5.2f}ms "
              f"inf={inference_ms:5.2f}ms total={total_ms:6.2f}ms{ekf_str}")

        self._writer.writerow([
            ts, step_id, f"{proba_high_risk:.4f}", flagged,
            predicted_class, f"{vision_ms:.2f}", f"{imu_ms:.2f}",
            f"{inference_ms:.2f}", f"{ekf_ms:.2f}", f"{total_ms:.2f}",
            f"{ekf_smooth:.4f}" if ekf_smooth is not None else "",
            ekf_alarm if ekf_alarm is not None else "",
            f"{ekf_rate:.4f}" if ekf_rate is not None else "",
        ])
        self._log_f.flush()

        if gpio_trigger and self.gpio_ready:
            threading.Thread(target=self._pulse_gpio, daemon=True).start()

    def _pulse_gpio(self, duration=0.6):
        try:
            GPIO = self.GPIO
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            GPIO.output(self.led_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            GPIO.output(self.led_pin, GPIO.LOW)
        except Exception as e:
            print(f"[GPIO] pulse failed ({e}) — continuing on console+CSV only")

    def close(self):
        self._log_f.close()
        if self.gpio_ready:
            try:
                self.GPIO.cleanup()
            except Exception:
                pass


# ============================================================
# LIVE CAMERA THREAD — just captures raw frames, no feature compute
# ============================================================
class CameraThread(threading.Thread):
    def __init__(self, camera_index, buffer_seconds=6.0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        self.buffer = deque()
        self.lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            now = time.monotonic()
            with self.lock:
                self.buffer.append((now, frame))
                cutoff = now - self.buffer_seconds
                while self.buffer and self.buffer[0][0] < cutoff:
                    self.buffer.popleft()

    def stop(self):
        self.running = False
        self.cap.release()


class ReplayCameraThread(threading.Thread):
    """Dry-run stand-in: replays a recorded video at real-world pace."""
    def __init__(self, video_path, buffer_seconds=6.0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.buffer = deque()
        self.lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.running = True

    def run(self):
        frame_interval = 1.0 / self.fps
        while self.running:
            t_frame_start = time.monotonic()
            ret, frame = self.cap.read()
            if not ret:
                print("[dry-run camera] end of video reached")
                break
            now = time.monotonic()
            with self.lock:
                self.buffer.append((now, frame))
                cutoff = now - self.buffer_seconds
                while self.buffer and self.buffer[0][0] < cutoff:
                    self.buffer.popleft()
            elapsed = time.monotonic() - t_frame_start
            time.sleep(max(0.0, frame_interval - elapsed))

    def stop(self):
        self.running = False
        self.cap.release()


# ============================================================
# VISION FEATURE THREAD — computes per-frame features ONCE, continuously,
# as new frames arrive. This is what makes short STEP_SEC affordable:
# the windowing step below only pools already-computed feature dicts,
# it never re-runs optical flow / Canny / etc on frames it's seen before.
# ============================================================
class VisionFeatureThread(threading.Thread):
    def __init__(self, camera_source, buffer_seconds=6.0, roi_scale=0.5):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.feat_buffer = deque()  # (ts, feats_dict)
        self.lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.roi_scale = roi_scale
        self.running = True
        self._last_ts = None
        self._prev_gray = None

    def run(self):
        while self.running:
            with self.camera_source.lock:
                snapshot = list(self.camera_source.buffer)
            new_frames = [
                (t, f) for (t, f) in snapshot
                if self._last_ts is None or t > self._last_ts
            ]
            new_frames.sort(key=lambda x: x[0])

            for ts, frame in new_frames:
                if self._prev_gray is None:
                    # first frame ever: compare against itself (near-zero flow),
                    # matches the original offline pipeline's first-frame handling.
                    # BUGFIX: must apply roi_scale here too, or the first flow
                    # call compares full-res prev_gray against a downscaled
                    # gray_roi and crashes with a shape-mismatch assertion.
                    _seed_roi = frame_road_roi(grey_world_wb(frame))
                    if self.roi_scale != 1.0:
                        _sh, _sw = _seed_roi.shape[:2]
                        _seed_roi = cv2.resize(
                            _seed_roi, (int(_sw * self.roi_scale), int(_sh * self.roi_scale)),
                            interpolation=cv2.INTER_AREA)
                    self._prev_gray = cv2.cvtColor(_seed_roi, cv2.COLOR_BGR2GRAY)
                feats, gray_roi = extract_frame_features(self._prev_gray, frame, roi_scale=self.roi_scale)
                with self.lock:
                    self.feat_buffer.append((ts, feats))
                    cutoff = time.monotonic() - self.buffer_seconds
                    while self.feat_buffer and self.feat_buffer[0][0] < cutoff:
                        self.feat_buffer.popleft()
                self._prev_gray = gray_roi
                self._last_ts = ts

            if not new_frames:
                time.sleep(0.005)  # idle briefly rather than busy-spin

    def get_window(self, t_start, t_end):
        with self.lock:
            return [f for (t, f) in self.feat_buffer if t_start <= t < t_end]

    def stop(self):
        self.running = False


# ============================================================
# LIVE IMU SERIAL THREAD — raw samples only; windowing/pooling is cheap
# numpy work done in the main loop, no need to precompute continuously.
# ============================================================
YPR_RE = re.compile(r"YPR\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)")
ACC_RE = re.compile(r"ACC\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)")


class IMUThread(threading.Thread):
    """
    ⚠️ VERIFY YPR_RE / ACC_RE against your actual firmware output before a
    live run (see earlier setup notes) — this assumes a combined
    '#YPR=y,p,r,ACC=ax,ay,az'-style line.
    """
    def __init__(self, port, baud=57600, buffer_seconds=6.0):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
        self.buffer = deque()
        self.lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.running = True
        self._last_ypr = None

    def run(self):
        while self.running:
            try:
                raw = self.ser.readline().decode(errors="ignore").strip()
            except Exception:
                continue
            if not raw:
                continue
            ypr_match = YPR_RE.search(raw)
            if ypr_match:
                self._last_ypr = tuple(float(x) for x in ypr_match.groups())
            acc_match = ACC_RE.search(raw)
            if acc_match and self._last_ypr is not None:
                ax, ay, az = (float(x) for x in acc_match.groups())
                yaw, pitch, roll = self._last_ypr
                now = time.monotonic()
                with self.lock:
                    self.buffer.append((now, ax, ay, az, roll, pitch, yaw))
                    cutoff = now - self.buffer_seconds
                    while self.buffer and self.buffer[0][0] < cutoff:
                        self.buffer.popleft()

    def get_window(self, t_start, t_end):
        with self.lock:
            return [r for r in self.buffer if t_start <= r[0] < t_end]

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


class ReplayIMUThread(threading.Thread):
    """Dry-run stand-in: replays a fusion CSV's IMU rows at real-world pace."""
    def __init__(self, fusion_csv, buffer_seconds=6.0):
        super().__init__(daemon=True)
        df = pd.read_csv(fusion_csv)
        df.columns = [c.upper() for c in df.columns]
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df = df.dropna(subset=["TIME"]).sort_values("TIME").reset_index(drop=True)
        self.imu = df[df["TYPE"] == "IMU"].reset_index(drop=True)
        required = ["AX", "AY", "AZ", "ROLL", "PITCH", "YAW"]
        missing = [c for c in required if c not in self.imu.columns]
        if missing:
            raise ValueError(f"Fusion CSV missing IMU columns: {missing}")
        self.buffer = deque()
        self.lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.running = True

    def run(self):
        if len(self.imu) < 2:
            return
        t0 = self.imu["TIME"].iloc[0]
        start_wall = time.monotonic()
        for _, row in self.imu.iterrows():
            if not self.running:
                break
            offset = (row["TIME"] - t0).total_seconds()
            target_wall = start_wall + offset
            sleep_for = target_wall - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            with self.lock:
                self.buffer.append((now, row["AX"], row["AY"], row["AZ"],
                                     row["ROLL"], row["PITCH"], row["YAW"]))
                cutoff = now - self.buffer_seconds
                while self.buffer and self.buffer[0][0] < cutoff:
                    self.buffer.popleft()
        print("[dry-run imu] end of fusion CSV reached")

    def get_window(self, t_start, t_end):
        with self.lock:
            return [r for r in self.buffer if t_start <= r[0] < t_end]

    def stop(self):
        self.running = False


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video", help="Required with --dry-run")
    ap.add_argument("--fusion-csv", help="Required with --dry-run")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--imu-port", default="/dev/ttyUSB0")
    ap.add_argument("--imu-baud", type=int, default=57600)
    ap.add_argument("--window-sec", type=float, default=2.0,
                     help="Trailing window span for feature pooling (must match your trained model)")
    ap.add_argument("--step-sec", type=float, default=0.5,
                     help="How often to emit a new score")
    ap.add_argument("--roi-scale", type=float, default=0.5,
                     help="RENAMED from --flow-scale. Downscales the whole ROI before "
                          "ALL vision features, not just flow -- measured on real Pi "
                          "hardware: flow-only downscaling gave just 1.28x speedup "
                          "because edge/texture stayed full-res. 1.0=full, 0.5=half, "
                          "0.25=quarter.")
    ap.add_argument("--gpio", action="store_true")
    ap.add_argument("--buzzer-pin", type=int, default=18)
    ap.add_argument("--led-pin", type=int, default=23)
    ap.add_argument("--log", default="skraid_live_log.csv")
    ap.add_argument("--max-steps", type=int, default=0,
                     help="Stop after N steps (0 = run until Ctrl+C / stream ends)")
    args = ap.parse_args()

    if args.dry_run and (not args.video or not args.fusion_csv):
        ap.error("--dry-run requires --video and --fusion-csv")
    if not (0.1 <= args.step_sec <= args.window_sec):
        ap.error(f"--step-sec should be between 0.1 and {args.window_sec} (your window span)")
    
    WINDOW_SEC = args.window_sec
    STEP_SEC = args.step_sec

    print(f"Loading model: {args.model}")
    model = joblib.load(args.model)
    feature_order = list(model.feature_names_in_)
    high_risk_idx = list(model.classes_).index("High Risk")
    print(f"Model ready — {len(feature_order)} features, threshold={HIGH_RISK_THRESHOLD}")
    print(f"Scoring cadence: every {STEP_SEC}s, pooled over trailing {WINDOW_SEC}s window")
    print(f"ROI downscale: {args.roi_scale}x -- applied to ALL vision features")

    buffer_seconds = WINDOW_SEC + 3.0

    if args.dry_run:
        print(f"*** DRY RUN — replaying {args.video} / {args.fusion_csv} ***")
        cam = ReplayCameraThread(args.video, buffer_seconds)
        imu = ReplayIMUThread(args.fusion_csv, buffer_seconds)
    else:
        print(f"*** LIVE — camera index {args.camera_index}, IMU {args.imu_port} ***")
        cam = CameraThread(args.camera_index, buffer_seconds)
        imu = IMUThread(args.imu_port, args.imu_baud, buffer_seconds)

    vision_feat = VisionFeatureThread(cam, buffer_seconds, roi_scale=args.roi_scale)
    cue = WarningCue(args.gpio, args.buzzer_pin, args.led_pin, args.log)

    # EKF is stateful across steps -- one instance for the whole run's
    # lifetime, created once here, never re-instantiated inside the loop.
    ekf = RiskEKF(dt=STEP_SEC, t_on=HIGH_RISK_THRESHOLD,
                  t_off=0.6 * HIGH_RISK_THRESHOLD)

    cam.start()
    imu.start()
    vision_feat.start()
    time.sleep(WINDOW_SEC + 0.5)  # let a full window's worth of data accumulate first

    step_id = 0
    t_next = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if t_next > now:
                time.sleep(t_next - now)
            t_now = time.monotonic()
            window_start = t_now - WINDOW_SEC
            window_end = t_now

            t_total_start = time.perf_counter()

            # ---- vision: pool already-computed per-frame features (cheap) ----
            t0v = time.perf_counter()
            frame_feats = vision_feat.get_window(window_start, window_end)
            if len(frame_feats) < 2:
                print(f"[step {step_id}] only {len(frame_feats)} vision samples in window — skipping")
                step_id += 1
                t_next += STEP_SEC
                if args.max_steps and step_id >= args.max_steps:
                    break
                continue
            vision_row = pool_vision_window(frame_feats)
            t1v = time.perf_counter()

            # ---- IMU: pool raw samples (cheap) ----
            t0i = time.perf_counter()
            imu_rows = imu.get_window(window_start, window_end)
            if len(imu_rows) < 10:
                print(f"[step {step_id}] only {len(imu_rows)} IMU samples in window — skipping")
                step_id += 1
                t_next += STEP_SEC
                if args.max_steps and step_id >= args.max_steps:
                    break
                continue
            imu_rows.sort(key=lambda r: r[0])
            ts = [r[0] for r in imu_rows]
            ax = np.array([r[1] for r in imu_rows])
            ay = np.array([r[2] for r in imu_rows])
            az = np.array([r[3] for r in imu_rows])
            roll = np.array([r[4] for r in imu_rows])
            pitch = np.array([r[5] for r in imu_rows])
            yaw = np.array([r[6] for r in imu_rows])
            dt = np.mean(np.diff(ts)) if len(ts) > 1 else (1.0 / IMU_RATE_ASSUMED)
            dt = dt if dt > 0 else (1.0 / IMU_RATE_ASSUMED)
            imu_row = extract_imu_window(ax, ay, az, roll, pitch, yaw, dt)
            t1i = time.perf_counter()

            # ---- inference ----
            t0m = time.perf_counter()
            merged = {**vision_row, **imu_row}
            x = pd.DataFrame([[merged[f] for f in feature_order]], columns=feature_order)
            proba = model.predict_proba(x)[0]
            predicted_class = model.classes_[np.argmax(proba)]
            p_high_risk = proba[high_risk_idx]
            flagged = p_high_risk >= HIGH_RISK_THRESHOLD
            t1m = time.perf_counter()

            # ---- EKF: adaptive-noise smoothing + hysteresis over p_high_risk ----
            t0e = time.perf_counter()
            R_k = tree_disagreement(model, x.values, high_risk_idx)
            ekf_out = ekf.step(p_high_risk, R_meas=R_k, dt=STEP_SEC)
            t1e = time.perf_counter()

            t_total_end = time.perf_counter()

            cue.fire(
                step_id=step_id,
                proba_high_risk=p_high_risk,
                flagged=bool(flagged),
                predicted_class=predicted_class,
                vision_ms=(t1v - t0v) * 1000,
                imu_ms=(t1i - t0i) * 1000,
                inference_ms=(t1m - t0m) * 1000,
                ekf_ms=(t1e - t0e) * 1000,
                total_ms=(t_total_end - t_total_start) * 1000,
                ekf_smooth=ekf_out.p_smooth,
                ekf_alarm=ekf_out.alarm,
                ekf_rate=ekf_out.p_rate,
            )

            step_id += 1
            t_next += STEP_SEC
            if args.max_steps and step_id >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    finally:
        cam.stop()
        imu.stop()
        vision_feat.stop()
        cue.close()
        print(f"Log written -> {args.log}")


if __name__ == "__main__":
    main()
