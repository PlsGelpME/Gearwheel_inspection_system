import cv2
import numpy as np
import matplotlib.pyplot as plt

# ── Load image and reuse gear centre from your existing code ───────────────
path = "images/endface1.png"
img  = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# ── Paste your existing centre detection here ──────────────────────────────
# (copy from find_centre.py — gear_cx, gear_cy, shaft_radius_px,
#  gear_radius_px, shaft_diameter_px, gear_diameter_px)
# For now we hardcode from your output — replace with your values
# ─────────────────────────────────────────────────────────────────────────
# Replace these with whatever your script printed:
gear_cx          = 573    # ← replace
gear_cy          = 420    # ← replace
shaft_radius_px  = 30.49     # ← rNeplace
gear_radius_px   = 402.465    # ← replace (tip circle radius)
# ─────────────────────────────────────────────────────────────────────────

# ── Step 1: High-resolution radial brightness profile ─────────────────────
# We scan from just outside the shaft hole to just beyond the tip circle.
# Use 720 rays (0.5° resolution) for accuracy.
# For each radius, sample 720 points around the full circle.

NUM_ANGLES  = 720
scan_start  = int(shaft_radius_px) + 5
scan_end    = int(gear_radius_px)  + 20   # slightly past tip

angles      = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
cos_angles  = np.cos(angles)
sin_angles  = np.sin(angles)

radii        = np.arange(scan_start, scan_end)
radial_mean  = []
radial_min   = []
radial_max   = []

for r in radii:
    xs = np.clip((gear_cx + r * cos_angles).astype(int), 0, w-1)
    ys = np.clip((gear_cy + r * sin_angles).astype(int), 0, h-1)
    vals = gray[ys, xs]
    radial_mean.append(vals.mean())
    radial_min.append(vals.min())
    radial_max.append(vals.max())

radial_mean = np.array(radial_mean)
radial_min  = np.array(radial_min)
radial_max  = np.array(radial_max)

# ── Step 2: Find tip, root and pitch radii ────────────────────────────────
# Working inward from the tip circle:
#
# The GEAR BODY region is between shaft and tip.
# Within the gear body the profile looks like:
#
#   brightness
#   ↑
#   │        /‾‾‾‾‾‾‾‾‾‾‾‾‾\   ← gear body (bright)
#   │       /               \
#   │______/   root circle   \_____  ← dark ring = tip circle
#   │        ↑               ↑
#   │     root dip         tip dip
#   └────────────────────────────→ radius
#
# The TIP dip is at gear_radius_px (already found).
# The ROOT dip is the local minimum INSIDE the gear body.
# The PITCH circle is halfway between root and tip.

# Focus on the gear body region only (inside the tip circle)
gear_body_mask = radii <= gear_radius_px
gear_body_mean = radial_mean[gear_body_mask]
gear_body_radii = radii[gear_body_mask]

# Find local minimum in gear body — that's the root circle dip
# Use scipy to find the dip closest to the tip
from scipy.signal import find_peaks, savgol_filter

# Smooth the profile to remove tooth-to-tooth noise
smoothed = savgol_filter(gear_body_mean, window_length=11, polyorder=3)

# Find valleys (minima) by inverting and finding peaks
valleys, props = find_peaks(-smoothed,
                             prominence=5,    # must be at least 5 brightness units deep
                             distance=10)     # valleys at least 10px apart

if len(valleys) > 0:
    # The root circle is the valley closest to the tip circle
    # (outermost valley in the gear body)
    root_idx        = valleys[np.argmax(gear_body_radii[valleys])]
    root_radius_px  = float(gear_body_radii[root_idx])
    root_brightness = float(gear_body_mean[root_idx])

    # Tooth depth = tip radius - root radius
    tooth_depth_px  = gear_radius_px - root_radius_px

    # Pitch radius = tip radius - addendum
    # For standard gears: addendum = module, dedendum = 1.25 module
    # tooth_depth = addendum + dedendum = 2.25 * module
    # So: addendum = tooth_depth / 2.25
    addendum_px     = tooth_depth_px / 2.25
    pitch_radius_px = gear_radius_px - addendum_px
    pitch_diameter_px = pitch_radius_px * 2

    print(f"── Radial profile analysis ──────────────────────────────")
    print(f"  Tip radius      : {gear_radius_px:.1f} px  (outer edge)")
    print(f"  Root radius     : {root_radius_px:.1f} px  (tooth root)")
    print(f"  Tooth depth     : {tooth_depth_px:.1f} px")
    print(f"  Addendum        : {addendum_px:.1f} px")
    print(f"  Pitch radius    : {pitch_radius_px:.1f} px")
    print(f"  Pitch diameter  : {pitch_diameter_px:.1f} px")
    print()
    print(f"  Tip/root ratio  : {gear_radius_px/root_radius_px:.4f}")
    print(f"  (standard gear) : 1.0556 expected  "
          f"(= (z+2)/(z-2.5×2/z) approx)")

else:
    print("No root circle valley found — try adjusting prominence")
    root_radius_px  = None
    pitch_radius_px = None
    pitch_diameter_px = None

# ── Step 3: Visualise ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Pitch circle detection", fontsize=13)

# Panel 1: radial profile with annotations
ax = axes[0]
ax.plot(gear_body_radii, gear_body_mean,
        color='steelblue', linewidth=1, alpha=0.5, label='raw mean')
ax.plot(gear_body_radii, smoothed,
        color='steelblue', linewidth=1.5, label='smoothed')

# Mark valleys
if len(valleys) > 0:
    ax.scatter(gear_body_radii[valleys], gear_body_mean[valleys],
               color='red', s=30, zorder=5, label='valleys')

# Vertical lines for key radii
ax.axvline(gear_radius_px, color='orange', linestyle='--',
           linewidth=1, label=f'tip r={gear_radius_px:.0f}px')

if root_radius_px:
    ax.axvline(root_radius_px, color='red', linestyle='--',
               linewidth=1, label=f'root r={root_radius_px:.0f}px')
    ax.axvline(pitch_radius_px, color='green', linestyle='--',
               linewidth=1, label=f'pitch r={pitch_radius_px:.0f}px')

ax.set_xlabel("Radius from gear centre (px)")
ax.set_ylabel("Mean brightness")
ax.set_title("Radial brightness profile\n"
             "(gear body region only)")
ax.legend(fontsize=8)

# Panel 2: three circles on original image
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Tip circle — orange
cv2.circle(vis, (gear_cx, gear_cy),
           int(round(gear_radius_px)), (0, 140, 255), 2)

if root_radius_px:
    # Root circle — red
    cv2.circle(vis, (gear_cx, gear_cy),
               int(round(root_radius_px)), (0, 0, 255), 2)
    # Pitch circle — green
    cv2.circle(vis, (gear_cx, gear_cy),
               int(round(pitch_radius_px)), (0, 255, 0), 2)

# Shaft hole — cyan
cv2.circle(vis, (gear_cx, gear_cy),
           int(round(shaft_radius_px)), (255, 255, 0), 2)

# Centre crosshair
cv2.line(vis, (gear_cx-30, gear_cy), (gear_cx+30, gear_cy), (255,255,255), 1)
cv2.line(vis, (gear_cx, gear_cy-30), (gear_cx, gear_cy+30), (255,255,255), 1)

# Legend text
font = cv2.FONT_HERSHEY_SIMPLEX
items = [
    ((0, 140, 255), f"Tip circle  d={gear_radius_px*2:.0f}px"),
    ((0, 255, 0),   f"Pitch circle d={pitch_diameter_px:.0f}px"
                    if pitch_diameter_px else "Pitch — not found"),
    ((0, 0, 255),   f"Root circle  d={root_radius_px*2:.0f}px"
                    if root_radius_px else "Root — not found"),
    ((255, 255, 0), f"Shaft hole   d={shaft_radius_px*2:.0f}px"),
]
for i, (col, label) in enumerate(items):
    y = 25 + i * 22
    cv2.rectangle(vis, (8, y-11), (22, y+3), col, -1)
    cv2.putText(vis, label, (28, y), font, 0.45, (0,0,0),       2)
    cv2.putText(vis, label, (28, y), font, 0.45, (255,255,255), 1)

axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[1].set_title("Three circles on gear image\n"
                   "orange=tip  green=pitch  red=root  cyan=shaft")
axes[1].axis('off')

plt.tight_layout()
plt.savefig("results/pitch_circle.png", dpi=150)
plt.show()
print("Saved to results/pitch_circle.png")