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

**Pi 4 vs. Pi 5, measured.** The same ROI-scale ablation was repeated on a
Raspberry Pi 5 against the same holdout session:

| ROI scale | Pi 4 | Pi 5 | Pi 5 vs. 2 s budget |
|---|---|---|---|
| `s=1.0` (full) | 70.05 s | 37.76 s | 18.9× over |
| `s=0.5` | 22.05 s | 12.12 s | 6.1× over |
| `s=0.25` | 11.92 s | 6.84 s | 3.4× over |

Pi 5 is consistently 1.74–1.86× faster at every scale, but its best
configuration is still 3.4× over budget. A full hardware-generation upgrade
closes less than half the gap — confirming the bottleneck is the
feature-extraction *algorithm class*, not the CPU generation.

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

**Why these weights (0.5 / 0.3 / 0.2)?** Set from exploratory analysis of
each term's mean contribution to the score (vision 0.259, IMU 0.089, gyro
0.087 — vision dominates), rather than by formal optimisation. This
ordering (vision > IMU > gyro) is preserved by the chosen weights while
damping vision's raw dominance. **The score is not sensitive to the exact
values**: perturbing each weight by ±0.05 and renormalising (27
combinations) changes the risk class for at most 5.1% of windows (mean
2.1%), with r ≥ 0.994 against the baseline score; even equal (1/3, 1/3,
1/3) weights agree with the chosen weighting on 88.3% of assignments
(r = 0.960). It's the *ordering* of the terms, not their precise values,
that carries the score. A systematic sensitivity analysis over the full
weight simplex remains future work.

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
├── dashboard/
│   ├── SKRAID_Dashboard.html      ← Public-facing risk map dashboard (open in browser)
│   └── sriad_points.js            ← Per-window GPS + risk score data, loaded by the dashboard
│
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

**GPS coverage.** Of the 42 sessions, 26 carry usable GPS. 39 sessions were
inventoried with GPS assessment (26 usable); the three August-2026
forward-holdout sessions used for classifier evaluation lost GPS fix
entirely while retaining valid IMU and camera data.

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

**Training details.** All architectures use `class_weight='balanced'` to
counter the 6.2% minority class (Gaussian NB excepted, as it admits no such
parameter). The reference RandomForest and Extra Trees use 200 estimators
at `max_depth=12`; the decision tree uses `max_depth=8`; LightGBM uses 200
estimators at `max_depth=8`. The threshold sweep spans 0.10–0.30.

**The collapse is not a sampling artefact of the holdout.** All nine
low-frame-rate sessions (see Limitations below) fall in the *training*
split; the three holdout sessions were captured at full 30 fps, so the
holdout features are free of that confound. The degradation occurs on
clean test data — any frame-rate artefact acts on the training side (24.8%
of training High-Risk windows), which would depress learned quality rather
than inflate the measured CV-to-holdout gap.

### Independent ground-truth check

54 spot-check windows labelled **cold** against raw video (no model output
shown):

| Comparison | Agreement with rider | Cohen's κ |
|---|---|---|
| `R_skid` formula vs. human | 42.6% | +0.067 |
| Trained RF (`t=0.15`) vs. human | 40.7% | +0.040 |

1 of 4 rider-confirmed High-Risk moments was caught. Reported as-is.

**Raw agreement overstates this.** Chance agreement alone is ≈0.38 on this
Safe-skewed three-class distribution, so both κ values are effectively
chance-level. The RF's confusion matrix is near row-identical to the
formula's, meaning the model stays bound to formula-derived label quality
rather than learning an independently better signal. Because neither the
score nor the video label is anchored to an actual friction measurement,
this near-zero κ shows the two *disagree* — it doesn't say which one is
closer to true traction. Resolving that needs an instrumented reference
(friction wheel or tyre-mounted sensing); we didn't have one for this
study. The result agrees in direction with the forward-holdout finding, so
two independent checks now concur that High-Risk detection is unresolved
at current data volume.

---

## Novelty & Positioning

SKRAID targets **skid risk**, not surface *type* — and reports the deployed
cost of doing so measured stage-by-stage on the target edge hardware,
rather than as model inference alone. Three things distinguish it from
prior work:

1. **Continuous, annotation-free score.** `R_skid` is computed directly
   from sensor statistics, so risk classes emerge from the score rather
   than from a human labelling surfaces — avoiding the annotation
   bottleneck that ties vision-only classifiers to pre-existing labelled
   corpora.
2. **Two-wheeler-specific sensing.** Roll and yaw rates enter the score as
   primary safety-relevant states, not suspension-damped disturbances the
   way they'd be treated on a four-wheeler.
3. **Indian road conditions.** The platform targets wet films, loose
   gravel, mud, and broken tar — conditions for which no annotated corpus
   exists — and contributes SRIAD to address that gap.

None of the surveyed literature (below) reports deployed latency broken
down by pipeline stage, nor validates against independent human judgement;
SKRAID addresses both.

## Contributions

1. **SRIAD**: 42 sessions / 8,546 windows / 285 min of synchronised
   camera + IMU + GPS data from real Indian road riding.
2. An **annotation-free continuous skid-risk score** (`R_skid`) fusing
   vision and inertial features over 2 s windows, GPS-tagged to road
   segments.
3. **Independent rider-confirmed ground-truth validation** on
   cold-labelled spot-check windows.
4. An **EKF over latent log-odds risk** providing bias-free, chatter-free
   temporal smoothing at negligible edge cost.
5. **Per-stage Pi 4 latency measurements** that localise the bottleneck,
   with implications for how edge-ADAS latency should be reported.
6. A **negative finding on cross-session generalisation** of discrete
   hazard classification, replicated across five model families.

## Related Work

- **Vision-based surface classification.** Venancio et al. benchmark five
  CNNs for 27-class road-surface classification on a Pi 4 for electric
  motorcycles (88.9% top-1 at 0.46 s/frame). Camera-only, and reliant on a
  960k-image annotated dataset that doesn't exist for Indian two-wheeler
  skid conditions.
- **IMU/vibration-based detection.** Mednis et al. detect potholes from
  Android accelerometer thresholds (~90% true-positive); Nericell monitors
  road and traffic conditions from smartphone sensors; Ikeda and Inoue
  report 86.6% within-subject but only 41.0% cross-subject accuracy — a
  generalisation gap analogous to ours. All lack visual context: an IMU
  can't perceive a wet patch before the wheel reaches it. Cui et al. apply
  random decision forests to pavement distress imagery, evaluated offline.
- **Multi-sensor fusion.** Ishizuki et al. fuse camera imagery with
  tire-mounted accelerometer and temperature features, improving on
  image-only classification but requiring instrumented tires and speeds
  above 30 km/h; Chu and Wu fuse roadside CCTV, weather, and tire-noise
  sensors. Both find vision weakest on wet surfaces, motivating SKRAID's
  inertial term.

See [References](#references) for full citations.

## Limitations

- **Not real-time as measured** — the system is a post-ride mapping tool,
  not an in-the-moment alert.
- **No friction ground truth**: `R_skid` is a proxy validated only by
  rider agreement and construct separability. Single vehicle, single
  rider, single region. GPS coverage on 26 of 42 sessions. Two night
  sessions excluded as outside the vision pipeline's operating envelope.
- `VIBRATION_ENERGY` and `ROUGHNESS_INDEX_IMU` use summation over the
  window, so their value scales with sample count (~35–45 at 20 Hz), an
  uncorrected sensitivity.
- Rider ground-truth sample is small (n=54, of which 4 High-Risk).
- **Frame rate is not uniform across the collection window**: nine of 42
  sessions (before a camera configuration change) pool only 5–10 frames
  per 2 s window against 120 for the majority — a 24× spread, at which
  dense optical flow's small-displacement assumption is violated at riding
  speed, and `p10`/`p90` pooling has little statistical basis. These
  sessions contribute 24.3% of all High-Risk windows, including the two
  most extreme (100% High-Risk) sessions — a possible confound between
  frame-rate artefact and hazard label that has not been isolated from
  genuine surface signal.

---

## Dashboard

`dashboard/SKRAID_Dashboard.html` is a self-contained, static risk map —
no server or build step needed. Open it directly in a browser.

It reads `sriad_points.js` (same folder) for every 2-second window's GPS
location, `R_skid` score, and risk class, and renders them as a heatmap and
clickable markers over the survey area (Pallavaram → IIIT Sri City).

**Seeing the actual road frame per point.** The dashboard doesn't ship with
the raw camera frames — they're too large for the repo. To see the actual
road image for any GPS-tagged point (potholes, wet patches, gravel, etc. as
the camera saw them):

1. Download the frames from Google Drive:
   [SKRAID captured frames](https://drive.google.com/drive/folders/1fjKn_jYO7Kz_EYQ7D9UqHm6ZXt0AZ62w?usp=sharing)
2. Open `SKRAID_Dashboard.html` in your browser.
3. Click **"Load Local Frames Folder"** near the top of the map and select
   the folder you just downloaded.
4. Click any marker on the map — its popup will now show the exact frame
   captured at that point, alongside its `R_skid` score and breakdown.

Frames are matched to markers by session and GPS coordinates, entirely in
the browser — nothing is uploaded anywhere. If you skip this step, popups
still work and show all scores, just without the frame image.

---

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

**EKF validation results.** The stage was validated in simulation before
deployment: unbiased step response, a single spurious impulse bounded to
four alarm steps, and zero alarm chatter when the score oscillates about
the threshold (the failure mode a fixed threshold exhibits). Measured cost
is ~50–100 ms per window, negligible beside the vision stage, so the layer
transfers unchanged to any future backend.

**Path to real-time (future work).**
Not a scale parameter — an architectural change. Either replace dense
Farnebäck flow with sparse Lucas–Kanade tracking over a bounded set of
corner features, or replace the six classical passes with a single
lightweight learned forward pass. Both require feature re-extraction and
retraining, with the accuracy cost *measured*, not assumed.

**Other directions.** Collecting more High-Risk exposure across varied
conditions to close the CV-to-holdout gap, and crowd-sourced aggregation
across multiple riders to build hazard maps denser than any single rider
can produce.

---

## Team

| Role | Name |
|------|------|
| Student | Adithya Ram S (S20230020276) |
| Guide | Dr. Anish Chand Turlapaty |
| Co-Guide | Prof. Hrishikesh Venkataraman |
| Institution | IIIT Sri City, Chittoor, Andhra Pradesh |

## Acknowledgement

The authors acknowledge the UKIERI-funded DIGIT Project (UK–India
Education and Research Initiative).

## References

1. R. Venancio et al., "Advanced driving assistance integration in electric
   motorcycles: road surface classification with a focus on gravel
   detection using deep learning," *Front. Artif. Intell.*, vol. 8, art.
   1520557, 2025.
2. A. Mednis et al., "Real time pothole detection using Android
   smartphones with accelerometers," in *Proc. DCOSS*, 2011, pp. 1–6.
3. P. Mohan, V. N. Padmanabhan, and R. Ramjee, "Nericell: rich monitoring
   of road and traffic conditions using mobile smartphones," in *Proc. ACM
   SenSys*, 2008, pp. 323–336.
4. L. Cui et al., "Pavement distress detection using random decision
   forests," in *Proc. ICDS*, LNCS vol. 9208, Springer, 2015, pp. 95–102.
5. Y. Ikeda and M. Inoue, "An estimation of road surface conditions using
   participatory sensing," in *Proc. IEEE GCCE*, 2017.
6. M. Ishizuki et al., "Image and CAIS features-based estimation of road
   surface condition on winter local road," *Proc. IEEE GCCE*, 2022,
   pp. 79–80.
7. X. Chu and Y. Wu, "Low cost road condition recognition based on
   roadside multi-sensors," *Proc. APCIP*, 2009, pp. 173–176.
8. S. M. Najib et al., "Road hazard detection for the motorcycle based on
   EfficientNet-Lite0," *Proc. IEEE ISCAIE*, 2023, pp. 101–105.
9. H. Venkataraman et al., "STRC@IIITS Driving Dataset for India (SIDDI),"
   STRG, IIIT Sri City, 2019–.
10. MoRTH, Govt. of India, "Road Accidents in India 2023," 2025.

---

## License

MIT — see [LICENSE](LICENSE).
