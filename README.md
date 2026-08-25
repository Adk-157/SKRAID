# SKRAID — Road Surface & Skid Risk Assessment for Two-Wheelers

> **B.Tech Honours Project** | ECE | IIIT Sri City
> Paper target: IEEE VTC

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is SKRAID?

SKRAID is a two-wheeler-mounted sensing system that computes a continuous
**skid risk score (`R_skid`)** per 2-second window by fusing three edge
sensors — camera, IMU, and GPS — on a Raspberry Pi 4, with **no manual
annotation** anywhere in the pipeline.

Scores are GPS-tagged and aggregated into **road-segment hazard maps** for
route planning and maintenance prioritisation.

> **SKRAID** = system name · **SRIAD** = dataset name (Skid Risk Indian
> Roads Dataset). Distinct, never interchangeable.

### ⚠️ On real-time operation — read this first

**SKRAID does not currently run in real time, and this repo does not claim
it does.** Measured end-to-end latency on the target Raspberry Pi 4:

| Configuration | Per-window latency | vs. 2 s budget |
|---|---|---|
| Full resolution | 70.05 s | 35.0× over |
| Downscale optical flow only (`--flow-scale 0.5`) | 33.30 s | 16.7× over |
| Downscale whole ROI (`--roi-scale 0.5`) | 22.05 s | 11.0× over |
| Downscale whole ROI (`--roi-scale 0.25`) | 11.92 s | 6.0× over |

Quarter scale is the practical floor before image content degrades past
usefulness. The bottleneck is **not** resolution but the algorithm class:
six classical per-pixel CV passes per frame (dense Farnebäck flow,
Canny+Sobel, double box-filter texture, HSV/LAB conversion) is too
expensive on this CPU at any usable resolution. An OpenCV build
misconfiguration was ruled out — the Pi's build has NEON_DOTPROD,
NEON_FP16, and ASM SGEMM/SGEMV kernels enabled.

Practically this means `live_deploy_skraid.py` **starves** when replaying
at real-world pace: the vision thread falls further behind every frame and
never fills a window. Use `benchmark_skraid_pi4.py` for measurement and
offline scoring. Real-time alerting is scoped future work (see
[Known Design Decisions](#known-design-decisions)).

---

## R_skid Formula

```
R_skid = 0.5 × V_vision  +  0.3 × σ²(a_z)  +  0.2 × (Δroll + Δyaw)
```

| Score | Label |
|-------|-------|
| < 0.33 | 🟢 Safe |
| 0.33 – 0.66 | 🟡 Caution |
| ≥ 0.66 | 🔴 High Risk |

**Speed multiplier: evaluated and REJECTED.** Multiplying `R_skid` by a
GPS-speed factor demoted 1,840 / 4,645 GPS-covered windows (40%) — including
**157 of 160 High-Risk windows** — into the Safe class, because the class
thresholds were calibrated against the unscaled score and any multiplier
below 1.0 shifts the whole distribution down without recalibration.
`R_skid_final = R_skid_base`. Do not re-enable without recalibrating
thresholds.

**Normalisation**
- Vision → independent hazard sub-scores, percentile-scaled to [0,1]
  globally, combined worst-case. Already 0–1.
- IMU/gyro → **per-session** min-max (mounting bias makes global unfair;
  `ACCEL_Z_VAR` spans 0.00006–17585 across sessions).
- No-IMU sessions (21% of windows) → vision-only score, rescaled to [0,1].

**Vision term — `V_vision = max(wet, rough, mud)`**

Road risk is **U-shaped in texture**: both an abnormally *smooth* surface
(water film mirrors light) and a *very rough* one (gravel, potholes) are
dangerous, while medium texture is a safe sealed road. A single averaged
term cannot represent both, so each hazard is scored separately and the
worst wins.

| Sub-score | Fires on | Signal |
|---|---|---|
| `wet` | water film / wet asphalt | `WETNESS_TEXTURE` (primary), smoothness + low-chroma composite (fallback) |
| `rough` | gravel / potholes | high `TEXTURE_ROUGHNESS` **or** `EDGE_DENSITY_MEAN` |
| `mud` | loose wet mud | high `MUD_SCORE` |

> **`WETNESS_RATIO` (HSV-threshold) is deliberately NOT used.** It
> correlates **−0.79** with truly wet roads — it keys on *brightness*, so
> over-exposed **dry** roads read "wet" (~0.9) while dark, genuinely **wet**
> asphalt reads "dry" (~0.04). No threshold tuning fixes a sign error.

---

## Hardware

| Component | Spec |
|-----------|------|
| Edge computer | Raspberry Pi 4 Model B — 8 GB |
| Camera | Logitech Brio 100 — 1280×720 @ 30 fps, MJPEG, `/dev/video0` |
| IMU | SparkFun 9-DoF Razor — `/dev/ttyUSB0`, 57600 baud, ~20 Hz |
| GPS | LC76G GPS HAT — `/dev/serial0`, 9600 baud, 1 Hz, NMEA RMC |
| Platform | Two-wheeler (petrol scooter) |

---

## Repository Structure

```
SKRAID/
├── pi/
│   ├── collect_data.py            ← Data collection (runs on Pi)
│   ├── benchmark_skraid_pi4.py    ← Offline latency benchmark + scoring
│   ├── live_deploy_skraid.py      ← Live/replay inference loop (see latency note)
│   ├── skraid_ekf.py              ← EKF temporal smoothing module
│   └── requirements.txt
│
├── colab/
│   └── honours_fixed.ipynb        ← Full analysis pipeline — canonical
│
├── LICENSE                        ← MIT
└── README.md
```

---

## Dataset — SRIAD

**Skid Risk Indian Roads Dataset**

| Attribute | Value |
|-----------|-------|
| Sessions | 42 |
| Total duration | 284.9 minutes |
| Windows (2 s) | 8,546 |
| Road types | Dry asphalt, wet asphalt, wet mud, gravel, potholes |
| Location | IIIT Sri City surroundings, Chittoor district, AP |
| Availability | Released as a subset of the [SIDDI collection](https://drive.google.com/drive/folders/1roseEnOTW7dPQ_dtyIUZb5bVYuCyiRKW) |

Raw data (videos, fusion CSVs) is **not** in this repo. Contact the Smart
Transportation Research Group at IIIT Sri City for access.

### Sensor CSV formats (3 generations)

`normalize_columns()` (Cell 3) unifies all three:

| Generation | Files | Layout |
|---|---|---|
| Jul–Nov 2025 | `gps_*.csv` + `imu_*.csv` | Separate files, no `FRAME_ID` |
| Dec 2025 | `sensor_*.csv` | `TYPE, TIME, D1…D9` |
| Jan 2026+ | `fusion_*.csv` | Named columns incl. `SPEED_KMH`, `FRAME_ID` |

> **GPS speed defect (found and corrected).** The logger polls the LC76G
> ~3× faster than its true fix rate, so consecutive rows repeat identical
> fixes. A duplicate-detection heuristic misread these as *stationary*,
> zeroing derived speed for most windows even during genuine 20–60 km/h
> riding. Cell 9 now **deduplicates to distinct fixes** before computing
> Haversine speed (validated against a simulated 30 km/h trace with matched
> 3× oversampling: 29.47 km/h recovered vs. 26.22 km/h under the old logic).

---

## Results

Latest full pipeline run — 42 sessions / 8,546 windows:

| Risk class | Windows | Share |
|---|---|---|
| 🟢 Safe | 5,596 | 65.5% |
| 🟡 Caution | 2,419 | 28.3% |
| 🔴 High Risk | 531 | 6.2% |

Mean term contributions: vision 0.259, IMU 0.089, gyro 0.087 — vision
dominates.

### Classification: honest generalisation finding

Discretising `R_skid` and training five classifiers, with **three August-2026
sessions held out entirely** from training and tuning (546 windows, 10 true
High Risk):

| Model | CV HR Recall | Holdout HR Recall |
|---|---|---|
| Decision Tree (d=8) | 0.551 ± 0.240 | 0.10 |
| Extra Trees | 0.505 ± 0.346 | 0.00 |
| Gaussian NB | 0.000 ± 0.000 | 0.00 |
| LightGBM | 0.501 ± 0.313 | 0.10 |
| RandomForest | 0.425 ± 0.284 | **0.20** (at tuned `t=0.15`) |

**GroupKFold CV says 0.83 High-Risk recall. The true forward holdout says
0.20.** That gap is the finding, not noise. All five architectures collapse
the same way, which points at data scarcity (531 High-Risk windows are only
~229 *independent* hazard events) rather than architecture choice.
Accuracy near 0.80 alongside 0.00–0.20 recall is majority-class accuracy and
is not a meaningful headline metric.

### Independent ground-truth check

54 spot-check windows labelled **cold** against raw video (no model output
shown):

| Comparison | Agreement with rider |
|---|---|
| `R_skid` formula vs. human | 42.6% |
| Trained RF (`t=0.15`) vs. human | 40.7% |

1 of 4 rider-confirmed High-Risk moments was caught. Reported as-is.

---

## Quick Start (Pi)

```bash
git clone https://github.com/Adk-157/SKRAID.git
cd SKRAID

sudo apt install ffmpeg -y
pip install -r pi/requirements.txt --break-system-packages

# Collect a session (Ctrl+C to stop)
python3 pi/collect_data.py
```

Outputs land in `data_collection/`: `fusion_TIMESTAMP.csv`,
`video_TIMESTAMP.mp4`, `gps_frame_manifest_TIMESTAMP.csv`.
Upload to Drive, then run the Colab notebook.

### Benchmarking

```bash
python3 pi/benchmark_skraid_pi4.py \
    --model rf_model.joblib \
    --video  data_collection/video_TIMESTAMP.mp4 \
    --fusion-csv data_collection/fusion_TIMESTAMP.csv \
    --n-windows 10 \
    --roi-scale 0.25 \
    --out results.csv
```

> `--roi-scale` **must match the scale the model was trained at.** It
> downscales the ROI before *every* vision feature (flow, edge, texture,
> HSV, colour), so changing it shifts every feature's distribution — not
> just optical flow's. Mismatch = silent train/serve skew.

---

## Known Design Decisions

**Why per-session IMU normalisation?**
Mounting position varies across sessions, so absolute acceleration ranges
differ by orders of magnitude. Global normalisation would make mild
vibration look identical to severe potholes.

**Why global vision normalisation?**
Vision features need cross-session comparability — a dry-road session must
score lower than a wet-mud session on the same scale.

**Why is `ACCEL_Z_VAR` only available for Jan 2026+ sessions?**
The IMU output format changed. Older sessions lack the column; the pipeline
handles this with NaN-aware averaging, and no-IMU sessions fall back to the
vision-only score.

**Why an *Extended* Kalman filter and not a plain KF?**
Risk is a probability bounded on [0,1]; a linear-Gaussian filter would
predict out-of-range values and mis-state innovation variance near the
boundaries. `skraid_ekf.py` tracks latent risk in unbounded **log-odds**
space with a logistic measurement model — nonlinear, hence *extended*.
Measurement noise adapts per window from RandomForest tree disagreement,
and hysteresis (dual thresholds + minimum hold) prevents alarm chatter.

**Path to real-time (future work).**
Not a scale parameter — an architectural change. Either replace dense
Farnebäck flow with sparse Lucas–Kanade tracking over a bounded set of
corner features, or replace the six classical passes with a single
lightweight learned forward pass. Both require feature re-extraction and
retraining, with the accuracy cost *measured*, not assumed.

---

## Team

| Role | Name |
|------|------|
| Student | Adithya Ram S (S20230020276) |
| Guide | Dr. Anish Chand Turlapaty |
| Co-Guide | Prof. Hrishikesh Venkataraman |
| Institution | IIIT Sri City, Chittoor, Andhra Pradesh |

---

## License

MIT — see [LICENSE](LICENSE).
