import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = "images/endface1.png"
img  = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# ── Take horizontal and vertical line scans through image centre ───────────
# This shows us the exact brightness profile across the image
# so we can see the gear / dark ring / reflection structure clearly

cx = w // 2
cy = h // 2

horiz_scan = gray[cy, :]          # middle row
vert_scan  = gray[:, cx]          # middle column

# ── Also take a radial scan from image centre outward ─────────────────────
# This is the most informative — shows brightness vs distance from centre
max_radius = min(cx, cy)
radii      = np.arange(0, max_radius)
radial_scan = []

for r in radii:
    # Sample 36 points around a circle of this radius
    angles  = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    xs      = np.clip((cx + r * np.cos(angles)).astype(int), 0, w-1)
    ys      = np.clip((cy + r * np.sin(angles)).astype(int), 0, h-1)
    mean_val = gray[ys, xs].mean()
    radial_scan.append(mean_val)

radial_scan = np.array(radial_scan)

# ── Find the dark ring ─────────────────────────────────────────────────────
# The dark ring is where the radial scan has its minimum
# It separates the bright gear (inner) from bright reflections (outer)
dark_ring_radius = int(np.argmin(radial_scan))
dark_ring_value  = radial_scan[dark_ring_radius]

print(f"Image centre    : ({cx}, {cy})")
print(f"Dark ring radius: {dark_ring_radius} px")
print(f"Dark ring value : {dark_ring_value:.1f}  "
      f"(lower = darker ring)")
print(f"Gear centre val : {gray[cy, cx]:.1f}  "
      f"(should be bright)")
print(f"Outer val       : {radial_scan[-1]:.1f}  "
      f"(reflection brightness)")
print()
print(f"Contrast gear→ring  : "
      f"{gray[cy,cx] - dark_ring_value:.1f} px")
print(f"Contrast ring→outer : "
      f"{radial_scan[-1] - dark_ring_value:.1f} px")

# ── Visualise ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Brightness structure analysis — endface1", fontsize=13)

# Panel 1: original image with scan lines marked
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.line(vis, (0, cy), (w, cy), (0, 255, 0), 1)
cv2.line(vis, (cx, 0), (cx, h), (0, 255, 255), 1)
cv2.circle(vis, (cx, cy), dark_ring_radius, (0, 0, 255), 2)
axes[0][0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[0][0].set_title("Scan lines + dark ring (red circle)")
axes[0][0].axis('off')

# Panel 2: horizontal scan
axes[0][1].plot(horiz_scan, color='green', linewidth=0.8)
axes[0][1].axvline(cx, color='gray', linestyle='--', linewidth=0.8)
axes[0][1].axhline(dark_ring_value, color='red',
                   linestyle='--', linewidth=0.8,
                   label=f'dark ring={dark_ring_value:.0f}')
axes[0][1].set_title("Horizontal scan (green line)")
axes[0][1].set_xlabel("x pixel")
axes[0][1].set_ylabel("Brightness")
axes[0][1].legend()

# Panel 3: vertical scan
axes[1][0].plot(vert_scan, color='cyan', linewidth=0.8)
axes[1][0].axvline(cy, color='gray', linestyle='--', linewidth=0.8)
axes[1][0].axhline(dark_ring_value, color='red',
                   linestyle='--', linewidth=0.8,
                   label=f'dark ring={dark_ring_value:.0f}')
axes[1][0].set_title("Vertical scan (cyan line)")
axes[1][0].set_xlabel("y pixel")
axes[1][0].set_ylabel("Brightness")
axes[1][0].legend()

# Panel 4: radial scan — most important
axes[1][1].plot(radii, radial_scan, color='orange', linewidth=1)
axes[1][1].axvline(dark_ring_radius, color='red',
                   linestyle='--', linewidth=1,
                   label=f'dark ring r={dark_ring_radius}px')
axes[1][1].axhline(dark_ring_value, color='red',
                   linestyle=':', linewidth=0.8)
axes[1][1].fill_between(radii[:dark_ring_radius],
                         radial_scan[:dark_ring_radius],
                         alpha=0.2, color='blue', label='gear region')
axes[1][1].fill_between(radii[dark_ring_radius:],
                         radial_scan[dark_ring_radius:],
                         alpha=0.2, color='orange', label='background')
axes[1][1].set_title("Radial brightness profile\n"
                      "(most important — shows gear structure)")
axes[1][1].set_xlabel("Distance from image centre (px)")
axes[1][1].set_ylabel("Mean brightness")
axes[1][1].legend()

plt.tight_layout()
plt.savefig("results/profile_scan.png", dpi=150)
plt.show()