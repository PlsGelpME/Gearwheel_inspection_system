"""
pitch_circle.py
===============
Finds the pitch circle of a gear by scanning INWARD from the outer edge.

Concept
-------
Starting at the outer gear edge and moving toward the centre, the layers are:

  [outer dark ring]  →  [bright gear body]  ←── pitch circle is HERE
                               ↓
                        [inner dark ring / shaft hole]

The pitch circle is the FIRST circle (scanning inward from the outer edge)
where ALL 360 sampled points are at 70%+ brightness — i.e. we have just
entered the fully-bright gear body after crossing the outer dark ring.

Expected result: pitch diameter ≈ 70–80% of outer gear diameter.

Algorithm
---------
1.  Run GearAnalyser  →  gear centre, outer edge radius, corrected image.
2.  Use GearAnalyser.radial_scan(inward=True) with a brightness condition.
    Each ray fires from r_outer inward; the first bright pixel on every ray
    is that ray's "pitch edge" point.
3.  The median radius of all pitch-edge points is the pitch circle radius
    (robust to outliers from tooth gaps).
4.  Fit a precise circle to those points with GearAnalyser.fit_circle().
5.  Sanity-check: pitch diameter should be 70–80% of outer diameter.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gear_analyser import GearAnalyser


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_PATH = "images/endface1.png"

# A pixel counts as "bright" (inside gear body) if its value ≥ this
BRIGHTNESS_FRACTION  = 0.70                          # 70 % of 255
BRIGHTNESS_THRESHOLD = int(round(BRIGHTNESS_FRACTION * 255))   # = 178

# Angular resolution for the inward scan (one ray per degree)
NUM_ANGLES = 360

# After collecting per-ray hit radii, ignore rays whose radius deviates
# more than this many px from the median (removes tooth-gap outliers)
OUTLIER_TOLERANCE_PX = 15


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Run GearAnalyser (shaft hole + perspective correction + outer edge)
# ══════════════════════════════════════════════════════════════════════════════

print("── Step 1: GearAnalyser ─────────────────────────────────────────────")
ga = GearAnalyser(IMAGE_PATH).run()
ga.print_summary()

img_work        = ga.corrected_gray          # perspective-corrected grayscale
cx, cy          = ga.gear_cx, ga.gear_cy
r_outer         = int(ga.gear_radius_px)     # scan starts HERE (outer edge)
r_inner_limit   = int(ga.shaft_radius_px) + ga.gap_beyond_shaft  # scan stops HERE


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Inward radial scan: first bright pixel on each ray
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each ray travels:  r_outer  →→→ (inward) →→→  r_inner_limit
#
#  Pixel sequence along a ray (outer → inner):
#    dark  dark  dark  | BRIGHT BRIGHT BRIGHT ...  (shaft hole dark)
#                      ↑
#               first bright pixel = pitch edge for this ray

print(f"\n── Step 2: Inward radial scan ───────────────────────────────────────")
print(f"  Brightness threshold : ≥ {BRIGHTNESS_THRESHOLD}  "
      f"({BRIGHTNESS_FRACTION*100:.0f}% of 255)")
print(f"  Scan range           : r={r_outer} → r={r_inner_limit}  (inward)")
print(f"  Rays                 : {NUM_ANGLES}")

pitch_points_raw, pitch_radii_raw = GearAnalyser.radial_scan(
    image        = img_work,
    cx           = cx,
    cy           = cy,
    scan_start_r = r_inner_limit,   # lower bound of search
    max_r        = r_outer,         # upper bound — scan STARTS here when inward=True
    num_angles   = NUM_ANGLES,
    condition    = lambda px: px >= BRIGHTNESS_THRESHOLD,
    inward       = True,            # ← scan from outer edge inward
)

print(f"  Raw hits             : {len(pitch_points_raw)} / {NUM_ANGLES}")

if len(pitch_points_raw) < 10:
    raise RuntimeError(
        "Too few bright hits — lower BRIGHTNESS_FRACTION or check image path")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Filter outliers (tooth gaps produce abnormally small radii)
# ══════════════════════════════════════════════════════════════════════════════

radii_arr  = np.array(pitch_radii_raw)
median_r   = float(np.median(radii_arr))
keep       = np.abs(radii_arr - median_r) <= OUTLIER_TOLERANCE_PX

pitch_points_filt = [p for p, k in zip(pitch_points_raw, keep) if k]
pitch_radii_filt  = radii_arr[keep]

n_removed = len(pitch_points_raw) - len(pitch_points_filt)
print(f"  Outliers removed     : {n_removed}  "
      f"(tolerance ±{OUTLIER_TOLERANCE_PX}px from median {median_r:.1f}px)")
print(f"  Clean hits           : {len(pitch_points_filt)}")

if len(pitch_points_filt) < 3:
    raise RuntimeError("Too few points after filtering — raise OUTLIER_TOLERANCE_PX")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Fit circle to filtered pitch-edge points
# ══════════════════════════════════════════════════════════════════════════════

pitch_cx, pitch_cy, pitch_radius_fit = GearAnalyser.fit_circle(pitch_points_filt)
pitch_diameter_fit = pitch_radius_fit * 2

# Ratio check
ratio = pitch_diameter_fit / ga.gear_diameter_px
print(f"\n── Step 4: Results ──────────────────────────────────────────────────")
print(f"  Median hit radius    : {median_r:.1f} px")
print(f"  Fitted pitch radius  : {pitch_radius_fit:.2f} px")
print(f"  Fitted pitch diameter: {pitch_diameter_fit:.2f} px")
print(f"  Fitted centre        : ({pitch_cx:.1f}, {pitch_cy:.1f})")
print(f"  Pitch / outer ratio  : {ratio:.3f}  "
      f"({'OK ✓' if 0.65 <= ratio <= 0.85 else 'outside expected 0.65–0.85 range'})")

print(f"\n══ FINAL MEASUREMENTS ════════════════════════════════════════════════")
print(f"  Gear centre          : ({cx}, {cy})")
print(f"  Shaft hole diameter  : {ga.shaft_diameter_px:.2f} px")
print(f"  Pitch circle diameter: {pitch_diameter_fit:.2f} px")
print(f"  Outer gear diameter  : {ga.gear_diameter_px:.2f} px")
print(f"  Addendum depth       : {ga.gear_radius_px - pitch_radius_fit:.2f} px  "
      f"(outer edge → pitch circle)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("Gear pitch circle — inward scan from outer edge", fontsize=13)

# ── Panel 1: all circles on corrected image ───────────────────────────────
vis1 = ga.draw_base(use_corrected=True)

# Outer edge (blue)
vis1 = ga.draw_circle_overlay(
    vis1, ga.gear_edge_cx, ga.gear_edge_cy, ga.gear_radius_px,
    color=(255, 100, 0),
    label=f"Outer d={ga.gear_diameter_px:.1f}px")

# Pitch circle (magenta, thicker)
vis1 = ga.draw_circle_overlay(
    vis1, pitch_cx, pitch_cy, pitch_radius_fit,
    color=(255, 0, 255),
    label=f"Pitch d={pitch_diameter_fit:.1f}px",
    thickness=2)

# Shaft hole (yellow)
cv2.ellipse(vis1, ga.shaft_ellipse, (0, 255, 255), 1)

# Centre crosshair (red)
GearAnalyser._draw_crosshair(vis1, cx, cy, (0, 0, 255))

axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
axes[0].set_title(
    "All circles\n"
    "blue=outer  magenta=pitch  yellow=shaft  red=centre")
axes[0].axis('off')

# ── Panel 2: per-ray hit points coloured by kept/removed ─────────────────
vis2 = ga.draw_base(use_corrected=True)

# Draw ALL raw hit points: green=kept, red=outlier
for pt, r_val in zip(pitch_points_raw, pitch_radii_raw):
    kept   = abs(r_val - median_r) <= OUTLIER_TOLERANCE_PX
    color  = (0, 255, 0) if kept else (0, 0, 255)
    cv2.circle(vis2, pt, 3, color, -1)

# Draw fitted pitch circle
cv2.circle(vis2,
           (int(round(pitch_cx)), int(round(pitch_cy))),
           int(round(pitch_radius_fit)),
           (255, 0, 255), 2)

# Draw outer edge for reference
cv2.circle(vis2,
           (int(round(ga.gear_edge_cx)), int(round(ga.gear_edge_cy))),
           int(round(ga.gear_radius_px)),
           (255, 100, 0), 1)

GearAnalyser._draw_crosshair(vis2, cx, cy, (0, 0, 255))

axes[1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
axes[1].set_title(
    f"Per-ray hits (inward scan)\n"
    f"green=kept  red=outlier  magenta=fitted pitch")
axes[1].axis('off')

# ── Panel 3: histogram of per-ray hit radii ───────────────────────────────
axes[2].hist(radii_arr, bins=40, color='steelblue', edgecolor='white',
             linewidth=0.5, label='all rays')
axes[2].axvline(median_r, color='green', linewidth=1.5,
                linestyle='--', label=f'median {median_r:.1f}px')
axes[2].axvline(pitch_radius_fit, color='magenta', linewidth=1.5,
                linestyle='-', label=f'fitted {pitch_radius_fit:.1f}px')
axes[2].axvline(ga.gear_radius_px, color='blue', linewidth=1,
                linestyle=':', label=f'outer {ga.gear_radius_px:.1f}px')
axes[2].axvspan(median_r - OUTLIER_TOLERANCE_PX,
                median_r + OUTLIER_TOLERANCE_PX,
                alpha=0.15, color='green', label='keep band')
axes[2].set_xlabel("Hit radius (px)")
axes[2].set_ylabel("Ray count")
axes[2].set_title(
    f"Distribution of inward hit radii\n"
    f"pitch ≈ {ratio*100:.1f}% of outer diameter")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/pitch_circle.png", dpi=150)
plt.show()
print("\nSaved → results/pitch_circle.png")