#!/usr/bin/env python3
"""
SKRAID — Full-Pipeline Inference Latency Benchmark for Raspberry Pi 4
========================================================================
Measures three stages separately, per 2-second window, over N windows
drawn from a real recorded session:

  1. Vision feature extraction  (Cell 8 logic: optical flow, edge, HSV,
     wetness/mud, texture, color — per-frame, then percentile-pooled
     across ~60 frames into one window-level feature row)
  2. IMU feature extraction     (Cell 9 logic: accel/gyro stats over the
     same 2-second window)
  3. RF inference                (rf.predict_proba on the merged 31-feature
     vector, High Risk threshold = 0.15)

Reports mean / median / p95 / max per stage, plus end-to-end total, so the
paper can cite a distribution rather than one cherry-picked number.

USAGE (run this ON the Pi, not on a dev machine):
    python3 benchmark_skraid_pi4.py \\
        --model rf_model.joblib \\
        --video /path/to/session_video.mp4 \\
        --fusion-csv /path/to/fusion_<session>.csv \\
        --n-windows 80 \\
        --out results_pi4.csv

Requires: opencv-python (or python3-opencv), pandas, numpy, scikit-learn,
joblib. See the PI SETUP CHECKLIST at the bottom of this file.
"""

import argparse
import time
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from skraid_ekf import RiskEKF, tree_disagreement

WINDOW_SEC = 2.0
IMU_RATE = 20.0
HIGH_RISK_THRESHOLD = 0.15  # decision threshold on predict_proba for 'High Risk'


# ------------------------------------------------------------------
# VISION — per-frame feature functions (mirrors Cell 8)
# ------------------------------------------------------------------
def grey_world_wb(frame_bgr):
    result = frame_bgr.copy().astype(np.float32)
    mean_b, mean_g, mean_r = (np.mean(result[:, :, i]) for i in range(3))
    mean_grey = (mean_b + mean_g + mean_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (mean_grey / (mean_b + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (mean_grey / (mean_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (mean_grey / (mean_r + 1e-6)), 0, 255)
    return result.astype(np.uint8)


def frame_road_roi(frame, top_ratio=0.4, bot_ratio=1.0):
    h, w = frame.shape[:2]
    return frame[int(h * top_ratio):int(h * bot_ratio), :]


def compute_optical_flow(prev_gray, curr_gray):
    """
    Farneback flow. Both inputs are expected ALREADY at the working
    resolution set by ROI_SCALE upstream -- this function no longer
    resizes anything itself. (Previously it downscaled internally, which
    only cut flow's own cost; edge/texture/color stayed full-res and
    became the new floor. Measured on real Pi hardware: downscaling flow
    alone gave only 1.28x end-to-end improvement despite flow itself
    being 4-33x faster in isolation.)
    """
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


def extract_frame_features(prev_gray, frame_bgr, top_ratio=0.4, roi_scale=0.5):
    """
    One frame's worth of vision features (all stages timed as a unit).

    ROI_SCALE FIX: resize happens ONCE here, on the cropped ROI, BEFORE any
    feature function runs -- flow, edge, texture, HSV, wetness, and color
    all operate on the SAME downscaled image. This replaces the earlier
    flow-only downscale, which left edge/texture/color at full resolution
    and made them the new bottleneck once flow itself was cut. White
    balance still runs on the full-resolution raw frame (cheap: it is
    just a per-channel mean and scale, not worth downscaling).

    roi_scale=1.0  : full resolution (original behaviour)
    roi_scale=0.5  : half resolution on EVERY feature, not just flow
    roi_scale=0.25 : quarter resolution on EVERY feature, not just flow
    """
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
    """
    Structural fix (percentile pooling): hazard-relevant features use
    percentile pooling instead of plain mean, so brief hazards (potholes,
    puddles <0.5s) aren't diluted across the ~60-frame window.
      EDGE_DENSITY_MEAN     -> p10 (a brief obstruction can locally REDUCE
                                edge density; low tail is the hazard signal)
      TEXTURE_ROUGHNESS     -> p10
      WETNESS_RATIO / MUD   -> p90 (a brief wet/mud patch is the max, not
                                the mean, of the wetness signal)
    Everything else: plain mean (context features).
    """
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


# ------------------------------------------------------------------
# IMU — per-window feature function (mirrors Cell 9)
# ------------------------------------------------------------------
def extract_imu_window(ax, ay, az, gx, gy, gz, dt):
    accel_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
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


# ------------------------------------------------------------------
# BENCHMARK DRIVER
# ------------------------------------------------------------------
def load_fusion_imu(fusion_csv):
    df = pd.read_csv(fusion_csv)
    df.columns = [c.upper() for c in df.columns]
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df.dropna(subset=["TIME"]).sort_values("TIME").reset_index(drop=True)
    imu = df[df["TYPE"] == "IMU"].copy()
    required = ["AX", "AY", "AZ", "ROLL", "PITCH", "YAW"]
    missing = [c for c in required if c not in imu.columns]
    if missing:
        raise ValueError(f"Fusion CSV missing IMU columns: {missing}")
    return imu


def run_benchmark(model_path, video_path, fusion_csv, n_windows, out_csv, roi_scale=0.5):
    print(f"Loading model: {model_path}")
    print(f"ROI downscale: {roi_scale}x -- applied to ALL vision features "
          f"(flow, edge, texture, HSV, color), not just optical flow.")
    if roi_scale == 0.5:
        print("  (1.0=full res, 0.5=half res on every feature, 0.25=quarter)")
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        model = joblib.load(model_path)
    feature_order = list(model.feature_names_in_)
    print(f"Model expects {len(feature_order)} features (using model's own order)")

    imu = load_fusion_imu(fusion_csv)
    t0 = imu["TIME"].iloc[0]
    dt = 1.0 / IMU_RATE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_per_window = int(round(fps * WINDOW_SEC))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_windows_video = total_video_frames // frames_per_window
    duration_imu = (imu["TIME"].iloc[-1] - t0).total_seconds()
    max_windows_imu = int(duration_imu // WINDOW_SEC)
    n_windows = min(n_windows, max_windows_video, max_windows_imu)
    if n_windows < 1:
        raise RuntimeError("Not enough synchronized data for even one window.")
    print(f"Benchmarking {n_windows} windows "
          f"(video cap: {max_windows_video}, IMU cap: {max_windows_imu})")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Empty video")
    # BUGFIX: prev_gray must be bootstrapped at the SAME resolution
    # extract_frame_features will produce internally, or the first flow
    # call in the loop below compares full-res prev_gray against a
    # downscaled gray_roi and crashes with a shape-mismatch assertion.
    _seed_roi = frame_road_roi(grey_world_wb(prev_frame))
    if roi_scale != 1.0:
        _sh, _sw = _seed_roi.shape[:2]
        _seed_roi = cv2.resize(_seed_roi, (int(_sw * roi_scale), int(_sh * roi_scale)),
                                interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(_seed_roi, cv2.COLOR_BGR2GRAY)

    vision_times, imu_times, inference_times, total_times = [], [], [], []
    ekf_times = []
    rows_log = []

    # EKF is stateful across windows -- one instance for the whole run, created
    # once outside the loop, not per-window.
    ekf = RiskEKF(dt=WINDOW_SEC, t_on=HIGH_RISK_THRESHOLD, t_off=0.6 * HIGH_RISK_THRESHOLD)

    for wid in range(n_windows):
        t_start_total = time.perf_counter()

        # ---- STAGE 1: vision ----
        t0v = time.perf_counter()
        frame_feats = []
        for _ in range(frames_per_window):
            ret, frame = cap.read()
            if not ret:
                break
            feats, prev_gray = extract_frame_features(prev_gray, frame, roi_scale=roi_scale)
            frame_feats.append(feats)
        if not frame_feats:
            break
        vision_row = pool_vision_window(frame_feats)
        t1v = time.perf_counter()
        vision_times.append(t1v - t0v)

        # ---- STAGE 2: IMU ----
        t0i = time.perf_counter()
        w_start = t0 + pd.Timedelta(seconds=wid * WINDOW_SEC)
        w_end = t0 + pd.Timedelta(seconds=(wid + 1) * WINDOW_SEC)
        w = imu[(imu["TIME"] >= w_start) & (imu["TIME"] < w_end)]
        if len(w) < 10:
            # BUGFIX: previously this `continue` fired after vision_times had
            # already been appended above, so skipped windows still polluted
            # the vision-stage stats while being absent from rows_log/out_csv.
            # Back that append out so timing stats and the logged row count
            # stay consistent.
            vision_times.pop()
            continue
        ax_, ay_, az_ = w["AX"].values, w["AY"].values, w["AZ"].values
        roll_diff = np.diff(w["ROLL"].values)
        pitch_diff = np.diff(w["PITCH"].values)
        yaw_diff = np.diff(w["YAW"].values)
        gx_ = np.pad(roll_diff / dt, (0, len(ax_) - len(roll_diff)), "edge")
        gy_ = np.pad(pitch_diff / dt, (0, len(ay_) - len(pitch_diff)), "edge")
        gz_ = np.pad(yaw_diff / dt, (0, len(az_) - len(yaw_diff)), "edge")
        imu_row = extract_imu_window(ax_, ay_, az_, gx_, gy_, gz_, dt)
        t1i = time.perf_counter()
        imu_times.append(t1i - t0i)

        # ---- MERGE + STAGE 3: inference ----
        merged = {**vision_row, **imu_row}
        x = pd.DataFrame([[merged[f] for f in feature_order]], columns=feature_order)

        t0m = time.perf_counter()
        proba = model.predict_proba(x)[0]
        high_risk_idx = list(model.classes_).index("High Risk")
        is_high_risk = proba[high_risk_idx] >= HIGH_RISK_THRESHOLD
        t1m = time.perf_counter()
        inference_times.append(t1m - t0m)

        # ---- EKF: adaptive-noise smoothing over the raw probability ----
        t0e = time.perf_counter()
        R_k = tree_disagreement(model, x.values, high_risk_idx)
        ekf_out = ekf.step(proba[high_risk_idx], R_meas=R_k, dt=WINDOW_SEC)
        t1e = time.perf_counter()
        ekf_times.append(t1e - t0e)

        t_end_total = time.perf_counter()
        total_times.append(t_end_total - t_start_total)

        rows_log.append({
            "window_id": wid,
            "vision_sec": t1v - t0v,
            "imu_sec": t1i - t0i,
            "inference_sec": t1m - t0m,
            "ekf_sec": t1e - t0e,
            "total_sec": t_end_total - t_start_total,
            "proba_high_risk": proba[high_risk_idx],
            "flagged_high_risk": is_high_risk,
            "ekf_smooth": ekf_out.p_smooth,
            "ekf_alarm": ekf_out.alarm,
            "ekf_rate": ekf_out.p_rate,
        })

        if (wid + 1) % 10 == 0:
            print(f"  window {wid + 1}/{n_windows} — "
                  f"total {rows_log[-1]['total_sec']*1000:.1f} ms")

    cap.release()

    def stats(arr, label):
        arr = np.array(arr) * 1000  # ms
        print(f"{label:22s} mean={arr.mean():8.2f} ms  "
              f"median={np.median(arr):8.2f} ms  "
              f"p95={np.percentile(arr, 95):8.2f} ms  "
              f"max={arr.max():8.2f} ms")

    print("\n" + "=" * 70)
    print(f"RESULTS — {len(rows_log)} windows successfully benchmarked")
    print("=" * 70)
    stats(vision_times, "Vision extraction")
    stats(imu_times, "IMU extraction")
    stats(inference_times, "RF inference")
    stats(ekf_times, "EKF smoothing")
    stats(total_times, "TOTAL per window")
    print("=" * 70)
    print(f"Window budget: {WINDOW_SEC*1000:.0f} ms  |  "
          f"Mean total: {np.mean(total_times)*1000:.2f} ms  |  "
          f"Headroom: {(WINDOW_SEC - np.mean(total_times))*1000:.2f} ms "
          f"({'OK — real-time capable' if np.mean(total_times) < WINDOW_SEC else 'OVER BUDGET'})")

    pd.DataFrame(rows_log).to_csv(out_csv, index=False)
    print(f"\nPer-window log saved -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to rf_model.joblib")
    ap.add_argument("--video", required=True, help="Path to a recorded session .mp4")
    ap.add_argument("--fusion-csv", required=True, help="Path to the matching fusion_<ts>.csv")
    ap.add_argument("--n-windows", type=int, default=80)
    ap.add_argument("--out", default="benchmark_results_pi4.csv")
    ap.add_argument("--roi-scale", type=float, default=0.5,
                     help="Downscale factor applied to the whole ROI before ALL vision "
                          "features (flow, edge, texture, HSV, color) -- not just optical "
                          "flow. RENAMED from --flow-scale: downscaling flow alone gave "
                          "only 1.28x real speedup on Pi hardware (measured) because "
                          "edge/texture stayed full-res and became the new bottleneck. "
                          "1.0=full res, 0.5=half, 0.25=quarter.")
    args = ap.parse_args()

    run_benchmark(
        model_path=args.model,
        video_path=args.video,
        fusion_csv=args.fusion_csv,
        n_windows=args.n_windows,
        out_csv=args.out,
        roi_scale=args.roi_scale,
    )


# ============================================================
# PI SETUP CHECKLIST (Debian 12 Bookworm, Pi 4 Model B, 8GB)
# ============================================================
# 1. Match scikit-learn version to whatever trained rf_model.joblib
#    (this environment shows it was pickled under sklearn 1.6.1 —
#    unpickling under a different version threw InconsistentVersionWarning).
#    On the Pi:
#        python3 -m venv ~/skraid-env
#        source ~/skraid-env/bin/activate
#        pip install --break-system-packages scikit-learn==1.6.1 joblib pandas numpy
#    If you don't know the exact training version, check it on Colab with:
#        import sklearn; print(sklearn.__version__)
#    and pin the Pi to match. A version mismatch usually just warns, but
#    tree-structure pickle format has changed across versions before —
#    don't find that out for the first time during the actual benchmark run.
#
# 2. OpenCV: prefer the apt package over pip on Pi/ARM — it's prebuilt for
#    the platform and avoids a slow from-source pip build:
#        sudo apt update && sudo apt install -y python3-opencv
#    If you need it inside the venv instead, use the headless pip wheel:
#        pip install --break-system-packages opencv-python-headless
#
# 3. Memory: a 200-tree, depth-12 RF is small (low tens of MB on disk);
#    8GB RAM is not a constraint here. The real memory/CPU pressure is
#    Farneback optical flow (compute_optical_flow) run per-frame across
#    ~60 frames/window — if vision-stage latency comes in high, that
#    function is the first place to profile/optimize (e.g. downsample
#    frame resolution before flow, or reduce Farneback pyramid levels).
#
# 4. Run this script itself with the venv's python3, using a real
#    recorded (video, fusion_csv) pair from an existing session so the
#    IMU and video streams are already time-aligned — no live scooter
#    run needed for this benchmark.
# ============================================================
