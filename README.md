# SKRAID — Real-Time Road Surface & Skid Risk Assessment for Two-Wheelers

> **B.Tech Honours Project** | ECE | IIIT Sri City  
> Target: IEEE INDICON / Springer ICCIDS

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red.svg)](https://www.raspberrypi.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![System](https://img.shields.io/badge/ADAS-L1%20Passive%20Warning-orange.svg)]()

---

## What is SKRAID?

SKRAID is an **L1 ADAS (passive warning) system** mounted on a two-wheeler that computes a per-window **skid risk score (R_skid)** in real-time from three edge sensors — camera, IMU, and GPS — without any cloud dependency for the alert itself.

A crowd-sourced cloud layer aggregates risk scores from multiple riders and serves a **GPS heatmap** to other users before they ride, optionally modulated by live weather data.

> **SKRAID** = System name  
> **SRIAD** = Dataset name (Skid Risk Indian Roads Dataset) — these are distinct, never interchangeable.

---

## System Architecture

```
┌─────────────────────────────── EDGE (Raspberry Pi 4) ──────────────────────┐
│                                                                              │
│  Logitech Brio 100  ──► Vision Features  ┐                                  │
│  (720p/30fps, MJPEG)    WETNESS_RATIO     │                                  │
│                         EDGE_DENSITY      │                                  │
│                         TEXTURE_ROUGHNESS ├──► R_skid Formula ──► LED/Buzzer │
│                         BRIGHTNESS_VAR    │                                  │
│                         MUD_SCORE         │                                  │
│                                           │                                  │
│  SparkFun 9-DoF IMU ──► IMU Features     │                                  │
│  (~20Hz, #YPR=...)      ACCEL_Z_VAR      │                                  │
│                         ROLL_RATE        │                                  │
│                         YAW_RATE         │                                  │
│                                           │                                  │
│  LC76G GPS HAT      ──► SPEED_KMH        ┘                                  │
│  (1Hz, NMEA RMC)                                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                              JSON per window
                     {lat, lon, r_skid, surface_type, ...}
                                    │
                                    ▼
┌──────────────── CLOUD (Flask + OpenWeatherMap) ─────────────────────────────┐
│  Aggregate crowd-sourced risk data                                           │
│  Weather modifier: Rain×1.6, Fog×1.4, Thunderstorm×2.0                      │
│  R_skid_displayed = min(1.0, R_skid_historical × modifier)                  │
│  ──► Leaflet.js GPS Risk Heatmap (pre-ride users)                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## R_skid Formula

```
R_skid_base  = 0.5 × V_vision
             + 0.3 × σ²(a_z)
             + 0.2 × (Δroll + Δyaw)

R_skid_final = R_skid_base × (1 + 0.3 × speed_norm)     # speed term currently DISABLED

speed_norm   = speed_kmh / 60.0   [clipped 0–1]
```

| Score | Label |
|-------|-------|
| < 0.33 | 🟢 Safe |
| 0.33 – 0.66 | 🟡 Caution |
| ≥ 0.66 | 🔴 High Risk |

> **Speed multiplier & motion gate are currently disabled** pending confirmation of the Haversine-derived speed (Cell 9). Until then `R_skid_final = R_skid_base.clip(0, 1)` — no speed weighting and no low-speed cap are applied.

**Normalization:**
- Vision features → **global** min-max (dry session 0.42 vs muddy 8.14 must be compared globally)
- IMU/Gyro features → **per-session** min-max (hardware mounting bias makes global unfair; ACCEL_Z_VAR ranges 0.00006–17585 across sessions)
- Sessions with no IMU data → `R_skid_base = V_vision` directly (`has_imu = (TERM_IMU > 0) | (TERM_GYRO > 0)`, no-IMU windows rescaled to full [0,1])

**Vision term features:** `WETNESS_RATIO`, `EDGE_DENSITY_MEAN`, `TEXTURE_ROUGHNESS`, `BRIGHTNESS_VARIANCE`, `MUD_SCORE` (optical flow excluded — see design notes).

---

## Hardware

| Component | Spec |
|-----------|------|
| Edge computer | Raspberry Pi 4 Model B — 8GB RAM |
| Camera | Logitech Brio 100 — 720p/30fps, MJPEG, `/dev/video0` |
| IMU | SparkFun 9-DoF Razor — `/dev/ttyUSB0`, 57600 baud, ~20Hz |
| GPS | LC76G GPS HAT — `/dev/serial0`, 9600 baud, 1Hz, NMEA RMC |
| Platform | Two-wheeler (petrol scooter) |

---

## Repository Structure

```
SKRAID/
├── pi/
│   ├── collect_data.py        ← Data collection script (runs on Pi)
│   └── requirements.txt       ← Pi Python dependencies
│
├── colab/
│   ├── honours_fixed.ipynb    ← Full analysis pipeline (Cells 0–12) — canonical
│   └── SKRAID_pipeline.ipynb  ← Markdown skeleton / cell outline
│
├── .gitignore                 ← Excludes raw data (AVI, CSV) from repo
├── LICENSE                    ← MIT
└── README.md
```

---

## Dataset — SRIAD

**Skid Risk Indian Roads Dataset**

| Attribute | Value |
|-----------|-------|
| Sessions | 33 (31 original + 2 collected 25 May 2026) |
| Total duration | ~223.7 minutes |
| Windows (2s) | 6,710 |
| Road types | Dry asphalt, wet mud, gravel, potholes |
| Location | IIIT Sri City surroundings, Chittoor district, AP |
| Drive | [Access via hvraman@iiits.in](https://drive.google.com/drive/folders/1Eg5fG3X_Rem8uqQID1Pa-6Fy4AsNK00O) |

> Raw data (AVI videos, fusion CSVs) is **not** in this repo. Contact the Smart Transportation Group at IIIT Sri City for access.

### Sensor CSV formats (3 generations)

The dataset spans three recording-script generations; `normalize_columns()` (Cell 3) unifies them:

| Generation | Files | Layout |
|-----------|-------|--------|
| Jul–Nov 2025 | `gps_*.csv` + `imu_*.csv` | Separate GPS/IMU files, no `FRAME_ID` |
| Dec 2025 | `sensor_*.csv` | `TYPE, TIME, D1…D9` (IMU `D1–D6`=AX,AY,AZ,ROLL,PITCH,YAW · GPS `D1,D2`=LAT,LON) |
| Jan 2026+ | `fusion_*.csv` | Named columns `AX,AY,AZ,ROLL,PITCH,YAW,LAT,LON,SPEED_KMH,FRAME_ID` |

> **Speed note:** the current collection script records `SPEED_KMH` (NMEA knots × 1.852), but across the existing 33 sessions GPS speed is largely absent or unreliable. The pipeline therefore **derives speed via the Haversine formula** from consecutive `LAT`/`LON` fixes in **Cell 9** rather than trusting the stored column.

---

## Colab Pipeline

The canonical pipeline is **`colab/honours_fixed.ipynb`** (13 cells, 0–12):

| Cell | Purpose |
|------|---------|
| Cell 0 | Mount Google Drive |
| Cell 1 | List sensor files (explorer) |
| Cell 2 | AVI → MP4 compression (H.264) |
| Cell 3 | `normalize_columns()` — maps old D1–D9 format to new AX/AY/AZ/ROLL/PITCH/YAW/LAT/LON/SPEED_KMH |
| Cell 4 | GPS Frame Extractor — seeks `FRAME_ID` in MP4, handles separate / sensor / fusion CSVs |
| Cell 5 | Heatmap window frame extractor (checkpoint: `len(existing) > 0`) |
| Cell 6 | Fusion CSV explorer + GPS Folium map (speed color-coded) |
| Cell 7 | pip install (opencv, pandas, numpy, tqdm, folium) |
| Cell 8 | Vision feature extraction per frame — grey-world white balance, `mud_score` (s<220), edge, HSV, texture |
| Cell 9 | IMU feature extraction per 2s window + **Haversine** GPS-speed derivation |
| Cell 10 | Merger — vision + IMU → label_helpers, with `recompute_wetness()` applied before aggregation + `brightness_variance`/`mud_score` passthrough |
| Cell 11 | R_skid computation + GPS heatmap + save dataset |
| Cell 12 | Analysis report + figures |

---

## Current Results

Latest pipeline run (post wetness-fix, speed multiplier disabled) over all 33 sessions / 6,710 windows:

| Risk class | Windows | Share |
|------------|---------|-------|
| 🟢 Safe | 4,998 | 74.5% |
| 🟡 Caution | 1,671 | 24.9% |
| 🔴 High Risk | 41 | 0.6% |

| Statistic | Value |
|-----------|-------|
| Mean R_skid | 0.261 |
| Max R_skid | 1.000 |
| Std R_skid | 0.133 |
| Mean term — Vision | 0.271 |
| Mean term — IMU | 0.057 |
| Mean term — Gyro | 0.030 |
| Windows with IMU | 1,801 (Jan 2026 + Apr/May 2026) |
| Windows without IMU | 4,909 (all 2025 sessions) |

> These numbers reflect the wetness-corrected run; they shift if the vision features are re-extracted or the speed multiplier is re-enabled.

---

## Quick Start (Pi Setup)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/SKRAID.git
cd SKRAID

# 2. Install system dependency
sudo apt install ffmpeg -y

# 3. Install Python packages
pip install -r pi/requirements.txt

# 4. Run a session
python pi/collect_data.py
# Press Ctrl+C to stop
```

Outputs land in `data_collection/`:
- `fusion_TIMESTAMP.csv`
- `video_TIMESTAMP.avi`
- `gps_frame_manifest_TIMESTAMP.csv`

Upload these to Google Drive, then run the Colab notebook.

---

## Known Design Decisions

**Why per-session IMU normalization?**  
The IMU mounting/handlebar position varies across sessions, so absolute acceleration ranges differ by orders of magnitude. Global normalization would make a session with mild vibration look identical to one with severe potholes. (Platform is a **petrol scooter** throughout — earlier "e-cycle" references were incorrect.)

**Why global vision normalization?**  
Vision features need cross-session comparability — a dry road session (WETNESS_RATIO ~0.42) must score lower than a wet mud session (~8.14) on the same scale.

**Why is ACCEL_Z_VAR only available for Jan 2026+ sessions?**  
The IMU output format was updated. Older sessions (D1–D9 / separate-file format) don't have this column. The pipeline handles this with NaN-aware averaging, and pre-2026 (no-IMU) sessions fall back to the vision-only score.

**Why is optical flow excluded from the vision term?**  
On a two-wheeler the camera suffers vibration/shake; optical flow in this setup reflects platform motion rather than surface properties, making it unreliable as a surface feature.

**Camera mount:** newer sessions use a **forward-pitch (dashcam-style)** mount — compatible with the RSCD reference dataset. Older sessions used a steeper downward angle, which is why those need per-session ROI crops (see the vision cell's ROI table).

**The wetness fix lives in the merger (Cell 10), not the vision cell.**  
The per-frame `wetness_ratio` from Cell 8 used loose HSV thresholds (`wet_tar s<50 v>120`) that flagged dry Indian concrete (saturation 8–25, value 130–150) as wet — wetness read 0.70–0.90 everywhere. `recompute_wetness()` re-derives it with tightened thresholds (`wet_tar s<30 v>180`, `wet_water s<15 v>200`, `wet_mud h:10–40 s:40–220 v:80–210`) **before** windowing, so Cell 8 does not need re-extraction. Three early high-glare sessions remain elevated — a documented limitation of passive monocular vision, not a bug.

---

## Team

| Role | Name |
|------|------|
| Student | Adithya Ram S (S20230020276) · [LinkedIn](https://linkedin.com/in/adithyarams) |
| Guide | Dr. Anish Chand Turlapaty · [LinkedIn](https://linkedin.com/in/anish-turlapaty-0ba33b18) |
| Co-Guide | Prof. Hrishikesh Venkataraman · [LinkedIn](https://linkedin.com/in/hrishikeshvenkataraman) |
| Institution | IIIT Sri City, Chittoor, Andhra Pradesh |

---

## Demo

[![Wet Road Skid Accident Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red)](https://youtube.com/shorts/xNVWfwQTG0I)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
