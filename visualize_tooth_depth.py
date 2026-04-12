import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.gear_core      import run_core_pipeline
from pipeline.gear_mask      import build_gear_mask
from pipeline.tooth_analysis import (get_contour_signal,
                                      detect_teeth_from_contour,
                                      measure_teeth_from_contour)

path      = "images/real_time_gear/rtg6.jpg"
PX_PER_MM = 20.0

m, inter = run_core_pipeline(path, px_per_mm=PX_PER_MM)
gray     = inter['gray']
gcx      = m['gear_centre_x']
gcy      = m['gear_centre_y']
tip_r    = inter['tip_radius_px']
root_r   = inter['root_radius_px']

gear_only, _ = build_gear_mask(
    gray, gcx, gcy, root_r * 0.15, tip_r)

angles, dists, pts = get_contour_signal(
    gear_only, gcx, gcy,
    tip_radius_px=tip_r)

tooth_peaks, gap_valleys, smoothed = detect_teeth_from_contour(
    dists, angles, expected_count=71)

measurements, summary = measure_teeth_from_contour(
    dists, angles, pts, tooth_peaks, gap_valleys,
    smoothed, gcx, gcy, PX_PER_MM)

# ── Build annotated image ──────────────────────────────────────────────────
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw all teeth
for mt in measurements:
    tip_pt   = (int(mt['tip_x']),   int(mt['tip_y']))
    left_pt  = (int(mt['left_x']),  int(mt['left_y']))
    right_pt = (int(mt['right_x']), int(mt['right_y']))

    cv2.circle(vis, tip_pt,   3, (0,   0,   255), -1)  # red   = tip
    cv2.circle(vis, left_pt,  3, (0,   255, 0  ), -1)  # green = left valley
    cv2.circle(vis, right_pt, 3, (255, 80,  0  ), -1)  # blue  = right valley
    cv2.line(vis, left_pt, right_pt, (255, 255, 0), 1)  # cyan  = tooth width

    # Tooth depth line: snapped tip to root circle surface
    tip_from_centre = np.sqrt(
        (mt['tip_x'] - gcx)**2 + (mt['tip_y'] - gcy)**2)
    root_r_fit = summary['root_radius_fitted_px']
    scale      = root_r_fit / tip_from_centre
    root_pt    = (int(gcx + (mt['tip_x'] - gcx) * scale),
                  int(gcy + (mt['tip_y'] - gcy) * scale))
    cv2.line(vis, tip_pt, root_pt, (255, 255, 255), 1)  # white = depth

# Draw fitted circles
tip_r_fit  = int(summary['tip_radius_fitted_px'])
root_r_fit = int(summary['root_radius_fitted_px'])
tip_fcx    = int(summary['tip_circle_cx'])
tip_fcy    = int(summary['tip_circle_cy'])
root_fcx   = int(summary['root_circle_cx'])
root_fcy   = int(summary['root_circle_cy'])

cv2.circle(vis, (tip_fcx,  tip_fcy),  tip_r_fit,  (0, 140, 255), 2)  # orange
cv2.circle(vis, (root_fcx, root_fcy), root_r_fit, (0,   0, 255), 2)  # red

# Pitch circle = tip - addendum = tip - depth/2.25
pitch_r_fit = int(tip_r_fit - (tip_r_fit - root_r_fit) / 2.25)
cv2.circle(vis, (tip_fcx, tip_fcy), pitch_r_fit, (0, 255, 0), 2)     # green

# Shaft hole
cv2.ellipse(vis, inter['shaft_ellipse'], (255, 255, 0), 2)

# Legend
font  = cv2.FONT_HERSHEY_SIMPLEX
items = [
    ((0,   0,   255), f"Tip points + circle  r={tip_r_fit}px"),
    ((0,   255, 0  ), f"Valley points + pitch r={pitch_r_fit}px"),
    ((0,   0,   200), f"Root circle  r={root_r_fit}px"),
    ((255, 255, 0  ), "Tooth width"),
    ((255, 255, 255), "Tooth depth"),
]
for i, (col, label) in enumerate(items):
    y = 28 + i * 24
    cv2.rectangle(vis, (8, y-12), (24, y+4), col, -1)
    cv2.putText(vis, label, (30, y), font, 0.45, (0,0,0),       2)
    cv2.putText(vis, label, (30, y), font, 0.45, (255,255,255), 1)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle("Fitted circles — tip, root, pitch", fontsize=13)

axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"All teeth annotated\n"
                   f"orange=fitted tip  red=fitted root  green=pitch")
axes[0].axis('off')

# Zoom top quadrant
r_int  = int(tip_r)
top    = max(0, gcy - r_int - 20)
bottom = min(gray.shape[0], gcy - int(root_r_fit * 0.6))
left   = max(0, gcx - int(tip_r * 0.6))
right  = min(gray.shape[1], gcx + int(tip_r * 0.6))
zoom   = vis[top:bottom, left:right]

if zoom.shape[0] > 10:
    axes[1].imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Zoomed — verify circles on teeth")
else:
    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Full view")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("results/fitted_circles.png", dpi=200)
plt.show()

print(f"\n── Fitted circle results ─────────────────────────────")
print(f"  Tip radius (fitted)  : {summary['tip_radius_fitted_px']:.1f} px")
print(f"  Root radius (fitted) : {summary['root_radius_fitted_px']:.1f} px")
print(f"  Pitch radius         : {pitch_r_fit} px")
print(f"  Tooth depth          : "
      f"{summary['tooth_depth_mean_px']:.1f} px  ",
      f"± {summary['tooth_depth_std_px']:.1f} px")
print(f"  Tooth width          : ",
      f"{summary['tooth_width_mean_px']:.1f} px  ",
      f"± {summary['tooth_width_std_px']:.1f} px")
print(f"  Gap width            : ",
      f"{summary['gap_width_mean_px']:.1f} px")