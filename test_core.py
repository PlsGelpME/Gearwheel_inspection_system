import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.gear_core import run_core_pipeline

# ── Set your px/mm ratio here ──────────────────────────────────────────────
# Arbitrary for now — change this when you have a calibration reference
PX_PER_MM = 20.0   # example: 20 pixels = 1 mm

path = "images/real_time_gear/rtg6.jpg"

print("Running core pipeline...")
print()

m, inter = run_core_pipeline(path, px_per_mm=PX_PER_MM)

print(f"── Measurements ─────────────────────────────────────────")
print(f"  Gear centre       : ({m['gear_centre_x']}, {m['gear_centre_y']})")
print(f"  Shaft diameter    : {m['shaft_diameter_px']:7.1f} px  "
      f"= {m['shaft_diameter_mm']:6.2f} mm")
print(f"  Tip diameter      : {m['tip_diameter_px']:7.1f} px  "
      f"= {m['tip_diameter_mm']:6.2f} mm")
print(f"  Root diameter     : {m['root_diameter_px']:7.1f} px  "
      f"= {m['root_diameter_mm']:6.2f} mm")
print(f"  Pitch diameter    : {m['pitch_diameter_px']:7.1f} px  "
      f"= {m['pitch_diameter_mm']:6.2f} mm")
print(f"  Tooth depth       : {m['tooth_depth_px']:7.1f} px  "
      f"= {m['tooth_depth_mm']:6.2f} mm")
print(f"  Circularity       : {m['circularity']:7.4f}")

# ── Annotated image ────────────────────────────────────────────────────────
gray  = inter['gray']
vis   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
gcx   = m['gear_centre_x']
gcy   = m['gear_centre_y']

# Draw all four circles
cv2.circle(vis, (gcx, gcy),
           int(inter['shaft_radius_px']),  (255, 255,   0), 2)  # cyan  = shaft
cv2.circle(vis, (gcx, gcy),
           int(inter['root_radius_px']),   (0,   0,   255), 2)  # red   = root
cv2.circle(vis, (gcx, gcy),
           int(inter['pitch_radius_px']),  (0,   255,   0), 2)  # green = pitch
cv2.circle(vis, (gcx, gcy),
           int(inter['tip_radius_px']),    (0,   140, 255), 2)  # orange= tip

# Edge points
for (ex, ey) in inter['edge_points'][::3]:
    cv2.circle(vis, (ex, ey), 1, (0, 255, 100), -1)

# Centre crosshair
cv2.line(vis, (gcx-40, gcy), (gcx+40, gcy), (255,255,255), 1)
cv2.line(vis, (gcx, gcy-40), (gcx, gcy+40), (255,255,255), 1)

# Legend
font  = cv2.FONT_HERSHEY_SIMPLEX
items = [
    ((0,   140, 255), f"Tip    {m['tip_diameter_mm']:.1f}mm"),
    ((0,   255,   0), f"Pitch  {m['pitch_diameter_mm']:.1f}mm"),
    ((0,   0,   255), f"Root   {m['root_diameter_mm']:.1f}mm"),
    ((255, 255,   0), f"Shaft  {m['shaft_diameter_mm']:.1f}mm"),
    ((0,   255, 100), f"Circ   {m['circularity']:.3f}"),
]
for i, (col, label) in enumerate(items):
    y = 28 + i * 24
    cv2.rectangle(vis, (8, y-12), (24, y+4), col, -1)
    cv2.putText(vis, label, (30, y), font, 0.5, (0,0,0),       2)
    cv2.putText(vis, label, (30, y), font, 0.5, (255,255,255), 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Core pipeline — all measurements", fontsize=13)

axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[1].set_title("Measurements overlay\n"
                   "orange=tip  green=pitch  red=root  cyan=shaft")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("results/core_output.png", dpi=150)
plt.show()
print("\nSaved to results/core_output.png")