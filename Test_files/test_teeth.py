import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.gear_core     import run_core_pipeline
from pipeline.tooth_analysis import (run_tooth_analysis,
                                      print_tooth_summary)

path      = "images/real_time_gear/rtg6.jpg"
PX_PER_MM = 20.0

# ── Core pipeline ──────────────────────────────────────────────────────────
m, inter = run_core_pipeline(path, px_per_mm=PX_PER_MM)

gear_cx      = m['gear_centre_x']
gear_cy      = m['gear_centre_y']
tip_r        = inter['tip_radius_px']
root_r       = inter['root_radius_px']
pitch_r      = inter['pitch_radius_px']
gray         = inter['gray']

# ── Tooth analysis ─────────────────────────────────────────────────────────
# Set expected_count=None for auto-detection first
# If count is wrong we'll set it to the known value
results, debug = run_tooth_analysis(
    gray, gear_cx, gear_cy,
    tip_r, root_r, pitch_r,
    px_per_mm=PX_PER_MM,
    expected_count=None
)

print_tooth_summary(results, PX_PER_MM)

# ── Plot ───────────────────────────────────────────────────────────────────
angles      = debug['angles']
signal      = debug['signal']
smoothed    = debug['smoothed']
threshold   = debug['threshold']
tooth_peaks = debug['tooth_peaks']
gap_valleys = debug['gap_valleys']

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle(f"Tooth detection — {results['tooth_count']} teeth found",
             fontsize=13)

# Panel 1: full 360° signal
axes[0].plot(angles, signal,   color='lightblue', linewidth=0.5,
             alpha=0.7, label='raw')
axes[0].plot(angles, smoothed, color='steelblue', linewidth=0.8,
             label='smoothed')
axes[0].axhline(threshold, color='orange', linestyle='--',
                linewidth=0.8, label=f'threshold={threshold:.0f}')
axes[0].scatter(angles[tooth_peaks], smoothed[tooth_peaks],
                color='red', s=8, zorder=5,
                label=f'{len(tooth_peaks)} teeth')
axes[0].set_title("Full 360° signal at tip-10% radius")
axes[0].set_xlabel("Angle (degrees)")
axes[0].set_ylabel("Brightness")
axes[0].set_xlim(0, 360)
axes[0].legend(fontsize=8)

# Panel 2: zoom — first 30°
zoom_mask = angles <= 30
axes[1].plot(angles[zoom_mask], signal[zoom_mask],
             color='lightblue', linewidth=0.8, alpha=0.7)
axes[1].plot(angles[zoom_mask], smoothed[zoom_mask],
             color='steelblue', linewidth=1.5)
axes[1].axhline(threshold, color='orange', linestyle='--', linewidth=0.8)
zoom_peaks = tooth_peaks[angles[tooth_peaks] <= 30]
axes[1].scatter(angles[zoom_peaks], smoothed[zoom_peaks],
                color='red', s=30, zorder=5)
axes[1].set_title("Zoomed — first 30° (verify peak placement)")
axes[1].set_xlabel("Angle (degrees)")
axes[1].set_ylabel("Brightness")
axes[1].set_xlim(0, 30)

# Panel 3: annotated gear image
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw sampling circle
cv2.circle(vis, (gear_cx, gear_cy),
           int(inter['tip_radius_px']), (0, 200, 255), 1)

# Mark each tooth peak on the image
pts_arr     = debug['pts']
tooth_peaks = debug['tooth_peaks']

for peak_idx in tooth_peaks:
    px = int(pts_arr[peak_idx][0])
    py = int(pts_arr[peak_idx][1])
    cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)

# Draw pitch circle
# True pitch circle from tooth analysis
true_pitch_r = int(debug['true_pitch_r'])
cv2.circle(vis, (gear_cx, gear_cy), 
           true_pitch_r, (0, 255, 0), 2)
cv2.circle(vis, (gear_cx, gear_cy),
           int(tip_r),   (0, 140, 255), 1)
cv2.circle(vis, (gear_cx, gear_cy),
           int(root_r),  (0, 0, 255), 1)

axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[2].set_title(f"Tooth peaks on image (red dots)\n"
                   f"cyan=sample circle  green=pitch  "
                   f"orange=tip  blue=root")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("results/tooth_output.png", dpi=150)
plt.show()

# ── Tooth count sanity check ───────────────────────────────────────────────
print(f"Expected tooth count : 71  (confirmed manually)")
print(f"Detected tooth count : {results['tooth_count']}")
if results['tooth_count'] != 71:
    diff = results['tooth_count'] - 71
    print(f"Difference           : {diff:+d}  "
          f"({'over-counting' if diff > 0 else 'under-counting'})")
    print(f"Suggestion           : set expected_count=71 to constrain detection")