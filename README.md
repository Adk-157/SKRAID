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

R_skid_final = R_skid_base × (1 + 0.3 × speed_norm)

speed_norm   = speed_kmh / 60.0   [clipped 0–1]
```

| Score | Label |
|-------|-------|
| < 0.33 | 🟢 Safe |
| 0.33 – 0.66 | 🟡 Caution |
| ≥ 0.66 | 🔴 High Risk |

**Normalization:**
- Vision features → **global** min-max (dry session 0.42 vs muddy 8.14 must be compared globally)
- IMU/Gyro features → **per-session** min-max (hardware mounting bias makes global unfair; ACCEL_Z_VAR ranges 0.00006–17585 across sessions)
- Sessions with no IMU data → `R_skid_base = V_vision` directly (rescaled to full [0,1])

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
│   ├── collect_data.py       ← Data collection script (runs on Pi)
│   └── requirements.txt      ← Pi Python dependencies
│
├── colab/
│   └── SKRAID_pipeline.ipynb ← Full analysis pipeline (Cells 0–11)
│
├── docs/
│   ├── SRIAD_README.docx     ← Dataset documentation
│   └── figures/              ← Representative result figures
│
├── .gitignore                ← Excludes raw data (AVI, CSV) from repo
└── README.md
```

---

## Dataset — SRIAD

**Skid Risk Indian Roads Dataset**

| Attribute | Value |
|-----------|-------|
| Sessions | 31 |
| Total duration | ~203 minutes |
| Windows (2s) | 6,102 |
| Road types | Dry asphalt, wet mud, gravel, potholes |
| Location | IIIT Sri City surroundings, Chittoor district, AP |
| Drive | [Access via hvraman@iiits.in](https://drive.google.com/drive/folders/1Eg5fG3X_Rem8uqQID1Pa-6Fy4AsNK00O) |

> Raw data (AVI videos, fusion CSVs) is **not** in this repo. Contact the Smart Transportation Group at IIIT Sri City for access.

---

## Colab Pipeline

| Cell | Purpose |
|------|---------|
| Cell 0 | Mount Google Drive |
| Cell 1 | AVI → MP4 compression (H.264) |
| Cell 2 | `normalize_columns()` — maps old D1–D6 format to new AX/AY/AZ/ROLL/PITCH/YAW/LAT/LON |
| Cell 3 | Fusion CSV explorer + visualizer |
| Cell 4 | GPS map (Folium, speed color-coded) |
| Cell 5 | pip install (opencv, pandas, numpy, tqdm) |
| Cell 6 | Vision feature extraction per frame (grey-world WB, mud_score, edge, HSV, texture) |
| Cell 7 | Vision windowing — aggregate frames → 2s windows |
| Cell 8 | IMU feature extraction per 2s window |
| Cell 9 | Merger — vision + IMU windows → label_helpers |
| Cell 10 | R_skid computation + GPS heatmap + save dataset |
| Cell 11 | Analysis report + figures |

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
The IMU is mounted differently across sessions (scooter vs e-cycle, different handlebar positions). Global normalization would make a session with mild vibration look identical to one with severe potholes because their absolute ranges differ by orders of magnitude.

**Why global vision normalization?**  
Vision features need cross-session comparability — a dry road session (WETNESS_RATIO ~0.42) must score lower than a wet mud session (~8.14) on the same scale.

**Why is ACCEL_Z_VAR only available for Jan 2026+ sessions?**  
The IMU output format was updated. Older sessions (D1–D6 format) don't have this column. The pipeline handles this with NaN-aware averaging.

**Why is optical flow excluded from the vision term?**  
The camera is mounted pointing downward at the road surface. Optical flow in this configuration reflects camera shake / IMU motion rather than surface properties, making it unreliable as a surface feature.

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
