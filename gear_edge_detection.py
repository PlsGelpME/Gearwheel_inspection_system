import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "images/endface1.png"
img  = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
img_cx, img_cy = w // 2, h // 2

#Helper
# ── Perspective correction helper ─────────────────────────────────────────
def correct_perspective(gray, ellipse):
    (cx, cy), (a, b), angle = ellipse
    if a < b:
        a, b = b, a
        angle += 90
    scale_y = a / b
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R     = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
    R_inv = R.T
    S     = np.array([[1.0, 0.0], [0.0, scale_y]])
    M2x2  = R_inv @ S @ R
    M = np.array([
        [M2x2[0,0], M2x2[0,1], cx - M2x2[0,0]*cx - M2x2[0,1]*cy],
        [M2x2[1,0], M2x2[1,1], cy - M2x2[1,0]*cx - M2x2[1,1]*cy]
    ])
    h, w = gray.shape
    corrected = cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    return corrected, M

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SHAFT HOLE / GEAR CENTRE  (your original code, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1: Search for darkest pixel near image centre ────────────────────
search_radius = int(min(w, h) * 0.30)

search_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(search_mask, (img_cx, img_cy), search_radius, 255, -1)

masked_gray = gray.copy()
masked_gray[search_mask == 0] = 255

min_val, _, min_loc, _ = cv2.minMaxLoc(masked_gray)
seed_x, seed_y = min_loc

print(f"Image size          : {w} x {h}")
print(f"Image centre        : ({img_cx}, {img_cy})")
print(f"Search radius       : {search_radius} px")
print(f"Darkest pixel       : ({seed_x}, {seed_y})  value={min_val}")

# ── Step 2-3: Threshold + connected component for shaft hole ──────────────
fill_tolerance  = 30
shaft_threshold = int(min_val) + fill_tolerance
_, shaft_mask   = cv2.threshold(
    gray, shaft_threshold, 255, cv2.THRESH_BINARY_INV)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    shaft_mask, connectivity=8)

seed_label = labels[seed_y, seed_x]

print(f"Connected components: {num_labels - 1}  (excluding background)")
print(f"Seed label          : {seed_label}")

if seed_label == 0:
    raise RuntimeError("Seed pixel is in background — adjust fill_tolerance")

shaft_component = np.zeros((h, w), dtype=np.uint8)
shaft_component[labels == seed_label] = 255

# ── Step 4: Morphology clean-up ───────────────────────────────────────────
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
shaft_clean = cv2.morphologyEx(
    shaft_component, cv2.MORPH_CLOSE, kernel, iterations=2)

# ── Step 5-6: Contour + ellipse fit → gear centre ─────────────────────────
contours, _ = cv2.findContours(
    shaft_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) == 0:
    raise RuntimeError("No shaft hole contour found")

shaft_contour = max(contours, key=cv2.contourArea)
shaft_area    = cv2.contourArea(shaft_contour)

if len(shaft_contour) < 5:
    raise RuntimeError("Shaft contour too small to fit ellipse")

ellipse = cv2.fitEllipse(shaft_contour)
(cx, cy), (axis1, axis2), angle = ellipse

gear_cx = int(round(cx))
gear_cy = int(round(cy))

shaft_diameter_px = (axis1 + axis2) / 2
shaft_radius_px   = shaft_diameter_px / 2

print(f"\nShaft hole found:")
print(f"  Area            : {shaft_area:.0f} px²")
print(f"  Centre          : ({gear_cx}, {gear_cy})")
print(f"  Diameter        : {shaft_diameter_px:.1f} px")
print(f"  Ellipse axes    : {axis1:.1f} x {axis2:.1f}")
print(f"  Ellipse angle   : {angle:.1f}°")
print(f"\nGear centre confirmed: ({gear_cx}, {gear_cy})")
print(f"Offset from image centre: "
      f"({gear_cx - img_cx:+d}, {gear_cy - img_cy:+d}) px")
corrected_gray, warp_M = correct_perspective(gray, ellipse)  # ← ADD THIS
tilt = np.degrees(np.arccos(min(axis1, axis2) / max(axis1, axis2)))
print(f"  Tilt estimate       : {tilt:.1f}°")               # ← ADD THIS


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — GEAR OUTER EDGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
#
# Layer structure radiating outward from centre:
#
#   [shaft hole – dark] → [gear body – bright] → [dark ring] → [reflection – bright]
#                                                      ↑
#                                                 gear EDGE
#
# Strategy:
#   • The darkest value inside the shaft hole is already known: min_val
#   • Cast NUM_ANGLES rays outward from gear centre.
#   • Skip pixels still inside/near the shaft hole.
#   • The first pixel on each ray whose brightness ≤ edge_threshold
#     is on the outer dark ring → that is the gear edge for that ray.
#   • Fit a circle to all collected edge points.

# ── Parameters (tune these if needed) ────────────────────────────────────
NUM_ANGLES       = 360   # angular resolution — 1 ray per degree
DARKNESS_MARGIN  = 15    # edge pixel must be ≤ (min_val + DARKNESS_MARGIN)
                         # ↑ raise if edge is missed on some rays
                         # ↓ lower if noisy inner rings trigger false hits
GAP_BEYOND_SHAFT = 10   # extra px beyond shaft_radius before scan starts
                         # prevents re-detecting the shaft wall
MAX_RADIUS       = int(min(w, h) * 0.55)  # hard upper limit for scan

# Threshold: a pixel is "dark enough to be the outer dark ring" if its
# brightness does not exceed this value.
# We reuse min_val (darkest pixel INSIDE shaft hole) from Part 1.
edge_threshold = int(min_val) + DARKNESS_MARGIN
scan_start_r   = int(shaft_radius_px) + GAP_BEYOND_SHAFT

print(f"\n── Gear edge detection ──────────────────────────────────────────")
print(f"  Shaft radius        : {shaft_radius_px:.1f} px")
print(f"  Scan starts at r    : {scan_start_r} px")
print(f"  Edge threshold      : ≤ {edge_threshold}  (shaft darkest={min_val})")
print(f"  Max search radius   : {MAX_RADIUS} px")

# ── Radial scan ───────────────────────────────────────────────────────────
angles      = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
edge_points = []    # (x, y) on the outer dark ring
edge_radii  = []    # distance from centre at which each edge was found

for angle in angles:
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for r in range(scan_start_r, MAX_RADIUS):
        x = int(round(gear_cx + r * cos_a))
        y = int(round(gear_cy + r * sin_a))

        if x < 0 or x >= w or y < 0 or y >= h:
            break                       # ray left the image

        if corrected_gray[y, x] <= edge_threshold:
            edge_points.append((x, y))
            edge_radii.append(r)
            break                       # edge found — next ray

print(f"  Edge points found   : {len(edge_points)} / {NUM_ANGLES}")

if len(edge_points) < 3:
    raise RuntimeError(
        "Too few edge points — increase DARKNESS_MARGIN or MAX_RADIUS")

# ── Circle fitting (algebraic least squares) ──────────────────────────────
def fit_circle_lstsq(points):
    """
    Fit a circle to (x,y) points using algebraic least squares.
    Solves:  2*cx*xi + 2*cy*yi + c = xi² + yi²
    Returns (cx, cy, radius).
    """
    pts        = np.array(points, dtype=np.float64)
    xi, yi     = pts[:, 0], pts[:, 1]
    A          = np.column_stack([2 * xi, 2 * yi, np.ones(len(xi))])
    b          = xi ** 2 + yi ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    ecx, ecy, c     = result
    radius          = np.sqrt(c + ecx ** 2 + ecy ** 2)
    return float(ecx), float(ecy), float(radius)

gear_edge_cx, gear_edge_cy, gear_radius_px = fit_circle_lstsq(edge_points)
gear_diameter_px = gear_radius_px * 2

print(f"\nGear outer edge:")
print(f"  Fitted centre       : ({gear_edge_cx:.1f}, {gear_edge_cy:.1f})")
print(f"  Radius              : {gear_radius_px:.1f} px")
print(f"  Diameter            : {gear_diameter_px:.1f} px")
print(f"  Median edge radius  : {np.median(edge_radii):.1f} px  "
      f"(std={np.std(edge_radii):.1f})")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — VISUALISATION  (4 panels)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle("Shaft hole → Gear centre → Gear outer edge", fontsize=13)

# ── Panel 1: search region & seed pixel ──────────────────────────────────
vis1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.circle(vis1, (img_cx, img_cy), search_radius, (0, 200, 255), 1)
cv2.circle(vis1, (seed_x, seed_y), 5, (0, 0, 255), -1)
cv2.putText(vis1, f"darkest ({seed_x},{seed_y})",
            (seed_x + 8, seed_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Search region (cyan)\nDarkest px = red dot  val={min_val}")
axes[0].axis('off')

# ── Panel 2: shaft hole mask ──────────────────────────────────────────────
axes[1].imshow(shaft_clean, cmap='gray')
axes[1].set_title(f"Shaft hole mask\nthreshold < {shaft_threshold}")
axes[1].axis('off')

# ── Panel 3: shaft contour + fitted ellipse + gear centre ────────────────
vis3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis3, [shaft_contour], -1, (0, 0, 255), 2)
cv2.ellipse(vis3, ellipse, (0, 255, 255), 2)
cv2.line(vis3, (gear_cx - 40, gear_cy), (gear_cx + 40, gear_cy), (0, 255, 0), 1)
cv2.line(vis3, (gear_cx, gear_cy - 40), (gear_cx, gear_cy + 40), (0, 255, 0), 1)
cv2.circle(vis3, (gear_cx, gear_cy), 4, (0, 255, 0), -1)
cv2.putText(vis3, f"Centre ({gear_cx},{gear_cy})",
            (gear_cx + 8, gear_cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(vis3, f"Shaft d={shaft_diameter_px:.1f}px",
            (gear_cx + 8, gear_cy + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
axes[2].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
axes[2].set_title("Shaft hole\nred=contour  yellow=ellipse  green=centre")
axes[2].axis('off')

# ── Panel 4: gear outer edge ──────────────────────────────────────────────
vis4 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Green dots  — individual radial edge points
for (ex, ey) in edge_points:
    cv2.circle(vis4, (ex, ey), 2, (0, 255, 0), -1)

# Blue circle — fitted gear outer edge
cv2.circle(vis4,
           (int(round(gear_edge_cx)), int(round(gear_edge_cy))),
           int(round(gear_radius_px)),
           (255, 100, 0), 2)

# Yellow ellipse — shaft hole reference
cv2.ellipse(vis4, ellipse, (0, 255, 255), 1)

# Red crosshair — gear centre
cv2.line(vis4, (gear_cx - 40, gear_cy), (gear_cx + 40, gear_cy), (0, 0, 255), 1)
cv2.line(vis4, (gear_cx, gear_cy - 40), (gear_cx, gear_cy + 40), (0, 0, 255), 1)
cv2.circle(vis4, (gear_cx, gear_cy), 4, (0, 0, 255), -1)

# Labels
lbl_y = int(gear_edge_cy) - int(gear_radius_px) - 10
cv2.putText(vis4, f"Gear d={gear_diameter_px:.1f}px",
            (int(gear_edge_cx) + 8, lbl_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
cv2.putText(vis4, f"Shaft d={shaft_diameter_px:.1f}px",
            (gear_cx + 8, gear_cy + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

axes[3].imshow(cv2.cvtColor(vis4, cv2.COLOR_BGR2RGB))
axes[3].set_title(
    "Gear outer edge\n"
    "green=edge pts  blue=fitted circle  yellow=shaft  red=centre")
axes[3].axis('off')

plt.tight_layout()
plt.savefig("results/gear_edge.png", dpi=150)
plt.show()
print("\nSaved to results/gear_edge.png")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n══ FINAL MEASUREMENTS ═══════════════════════════════════════════")
print(f"  Gear centre         : ({gear_cx}, {gear_cy})")
print(f"  Shaft hole diameter : {shaft_diameter_px:.2f} px")
print(f"  Gear outer diameter : {gear_diameter_px:.2f} px")
print(f"  Ratio (gear/shaft)  : {gear_diameter_px / shaft_diameter_px:.3f}")