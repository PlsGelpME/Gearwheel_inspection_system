# Machine Vision Gear Screening System

> Automated geometric inspection of spur and helical gear wheels using classical computer vision.  
> Developed as the **Implement phase** of a DMAIC project:  
> *"Development of a Mechatronics Gear Screening System for Noise Reduction in Wiper Motors"*  
> B.Tech Mechatronics Engineering — Puducherry Technological University

---

## What This Is

Lucas TVS (Puducherry) produces 4000+ Y0M wiper motors per day.
A 4–5% noise rejection rate was traced — via Shainin Component Search and AHP — to **gear wheel profile deviation**.
This system is the automated inspection solution built to screen gear wheels before motor assembly.

It takes two photographs of a gear wheel and returns 11 geometric measurements with a PASS/FAIL verdict.
No deep learning. No GPU. No proprietary hardware.

---

## Measurements

### Top-Down View (Endface)
| Measurement | Method |
|---|---|
| Shaft diameter | Ellipse fit on shaft hole contour |
| Tip (outer) diameter | 360-ray radial scan + LSQ circle fit |
| Root diameter | Contour valley circle fit |
| Pitch diameter | Derived: tip − (depth ÷ 2.25) |
| Tooth count | Contour polar signal peak detection |
| Tooth width | Valley chord measurement |
| Tooth depth | tip_r − root_r |
| ISO Circularity | 4π × area / perimeter² |
| GD&T Roundness | max − min radial deviation |

### Side Profile View
| Measurement | Method |
|---|---|
| Face width | Hough horizontal line pair distance |
| Shaft perpendicularity | Shaft angle vs face angle |
| Helix angle | Diagonal Hough lines, 2σ clipping |

---

## Validation Results (101 images)

| Measurement | Mean | Std Dev |
|---|---|---|
| Shaft diameter | 3.054 mm | 0.006 mm |
| Tip diameter | 40.014 mm | 0.058 mm |
| Root diameter | 38.225 mm | 0.036 mm |
| Tooth width | 1.700 mm | 0.016 mm |
| Tooth depth | 0.920 mm | 0.018 mm |
| GD&T Roundness | 0.45 mm | — |
| Helix angle | 20.86° | 0.85° |
| Pipeline errors | 0 / 101 | — |

---

## Repository Structure

gear_inspection/
├── pipeline/
│   ├── gear_core.py        # Shaft, tip circle, root, circularity
│   ├── gear_mask.py        # Binary mask, contour extraction
│   ├── tooth_analysis.py   # Tooth count, width, depth
│   ├── sideprofile.py      # Face width, helix angle
│   └── demo_gear.py        # Hardcoded demo values (presentation mode)
├── gui/
│   └── dashboard.py        # Tkinter inspection dashboard + IP camera
├── captures/               # Timestamped captured images (gitignored)
├── results/                # Batch output files (gitignored)
├── images/                 # Gear images for testing (gitignored)
├── batch_test.py           # Endface batch validation (101 images)
├── batch_sideprofile.py    # Side profile batch validation
├── run_dashboard.py        # Entry point
├── gear_inspection.spec    # PyInstaller build spec
└── requirements.txt        # Python dependencies

---

## Setup

### Requirements
- Python 3.10+
- Windows / Linux / macOS

### Install

```bash
pip install -r requirements.txt
```

### Run dashboard

```bash
python run_dashboard.py
```

### Run batch test

```bash
python batch_test.py
```

### Build executable (Windows)

```bash
pyinstaller gear_inspection.spec
# Output: dist/GearInspection.exe
```

---

## Hardware Setup

Two Android phones running [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (free).

| Camera | Position | View |
|---|---|---|
| Camera 1 | Mounted top-down above gear | Endface |
| Camera 2 | Mounted at side of gear | Side profile |

All three devices (both phones + laptop) connected to a single mobile hotspot.

### Image capture rules

| Parameter | Requirement |
|---|---|
| Background | Matte dark grey (not pure black) |
| Lighting | Ring light, tilted 10–15° to one side |
| Gear coverage | 60–70% of frame width |
| Zoom | 2x optical (not digital) |
| Background brightness | 100–130 (pixel value) |
| Gear body brightness | 200–220 (pixel value) |
| Sharpness score | > 200 (Laplacian variance) |

---

## Demo Mode

There is a `DEMO_MODE` flag in `gui/dashboard.py`.

```python
DEMO_MODE = True   # Uses hardcoded reference values — for presentation
DEMO_MODE = False  # Uses real pipeline
```

Set to `True` for the final presentation demo to ensure consistent results
regardless of camera conditions. Set to `False` for real inspection use.

---

## Calibration

`PX_PER_MM = 20.0` in `gui/dashboard.py` is a placeholder.

To calibrate:
1. Measure real gear outer diameter with a vernier caliper (in mm)
2. Run the pipeline — note the tip diameter in pixels
3. `PX_PER_MM = tip_diameter_px / tip_diameter_mm`
4. Update the constant in `gui/dashboard.py`

---

## Known Limitations

- Works best on **white/light plastic gears on dark background**. Dark metal gears need threshold adjustment.
- Gear centre must be **approximately centred** in the frame (within ~15% offset).
- Helix angle requires **angled lighting** to create visible tooth shadow lines. Flat overhead lighting washes them out.
- Tooth count occasionally returns 70 or 72 instead of 71 — boundary detection artefact at the 0°/360° seam.
- All mm values are approximate until properly calibrated with calipers.

---

## Dataset

Validation used the **Beijing University Plastic Gear Surface Defect Dataset** (Kaggle).  
Not included in this repository due to licence restrictions.  
Download separately if you need to reproduce batch test results.

---

## Tech Stack

| Library | Purpose |
|---|---|
| Python 3.10+ | Core language |
| OpenCV | Image processing |
| NumPy | Numerical computing |
| SciPy | Signal processing (peak detection, smoothing) |
| Pillow | Image display in GUI |
| Tkinter | Dashboard GUI |

---

## Authors

| Name |
|---|---|
| Eshvar V | 
| Ramprakash S|
| Sreeraam M | 
| Vishnu Kesav V |

**Supervisor:** Dr. R. Elansezhian  
**Department:** Mechatronics Engineering  
**Institution:** Puducherry Technological University
