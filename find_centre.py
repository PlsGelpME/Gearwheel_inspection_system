import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "images/endface1.png"
img  = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
img_cx, img_cy = w // 2, h // 2

# ── Step 1: Search for darkest pixel near image centre ─────────────────────
# We search within a circular region around the image centre.
# The shaft hole will be somewhere near the centre of the frame.
# Search radius = 30% of the smaller image dimension.
search_radius = int(min(w, h) * 0.30)

# Build a mask for the search region
search_mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(search_mask, (img_cx, img_cy), search_radius, 255, -1)

# Apply mask — pixels outside search region become 255 (bright)
# so they won't be picked as darkest
masked_gray = gray.copy()
masked_gray[search_mask == 0] = 255

# Find the darkest pixel within the search region
min_val, _, min_loc, _ = cv2.minMaxLoc(masked_gray)
seed_x, seed_y = min_loc

print(f"Image size          : {w} x {h}")
print(f"Image centre        : ({img_cx}, {img_cy})")
print(f"Search radius       : {search_radius} px")
print(f"Darkest pixel       : ({seed_x}, {seed_y})  value={min_val}")

# ── Step 2: Flood fill from darkest pixel to find full shaft hole ──────────
# Flood fill expands outward from the seed pixel, collecting all
# connected pixels within a brightness tolerance of the seed.
# This captures the entire shaft hole as one connected region.

# Threshold — fill pixels darker than seed_value + tolerance
fill_tolerance = 30   # px brightness range to include in fill

# Create binary mask — pixels darker than threshold belong to shaft hole
shaft_threshold = int(min_val) + fill_tolerance
_, shaft_mask   = cv2.threshold(
    gray, shaft_threshold, 255, cv2.THRESH_BINARY_INV)

# ── Step 3: Keep only the connected component containing the seed ──────────
# There may be other dark regions in the image.
# We only want the one that contains our seed point.
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    shaft_mask, connectivity=8)

# Find which label the seed pixel belongs to
seed_label = labels[seed_y, seed_x]

print(f"Connected components: {num_labels - 1}  (excluding background)")
print(f"Seed label          : {seed_label}")

if seed_label == 0:
    print("WARNING: seed pixel is in background — adjust fill_tolerance")
else:
    # Isolate just the shaft hole component
    shaft_component = np.zeros((h, w), dtype=np.uint8)
    shaft_component[labels == seed_label] = 255

    # ── Step 4: Clean up with morphology ──────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shaft_clean = cv2.morphologyEx(
        shaft_component, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Step 5: Find contour of shaft hole ────────────────────────────────
    contours, _ = cv2.findContours(
        shaft_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        print("No shaft hole contour found")
    else:
        shaft_contour = max(contours, key=cv2.contourArea)
        shaft_area    = cv2.contourArea(shaft_contour)

        # ── Step 6: Fit circle to shaft hole contour ───────────────────────
        # fitEllipse is more robust than minEnclosingCircle
        # for slightly imperfect holes
        if len(shaft_contour) >= 5:
            ellipse = cv2.fitEllipse(shaft_contour)
            (cx, cy), (axis1, axis2), angle = ellipse

            # Gear centre = shaft hole centre
            gear_cx = int(round(cx))
            gear_cy = int(round(cy))

            # Shaft diameter = average of ellipse axes
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

            # ── Visualise ──────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("Shaft hole detection → gear centre", fontsize=13)

            # Panel 1: search region + seed point
            vis1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis1, (img_cx, img_cy),
                       search_radius, (0, 200, 255), 1)
            cv2.circle(vis1, (seed_x, seed_y),
                       5, (0, 0, 255), -1)
            cv2.putText(vis1, f"darkest px ({seed_x},{seed_y})",
                        (seed_x + 8, seed_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            axes[0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Search region (cyan)\n"
                               f"Darkest pixel = red dot  val={min_val}")
            axes[0].axis('off')

            # Panel 2: shaft hole mask
            axes[1].imshow(shaft_clean, cmap='gray')
            axes[1].set_title(f"Shaft hole mask\n"
                               f"threshold < {shaft_threshold}")
            axes[1].axis('off')

            # Panel 3: result on original image
            vis3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Draw shaft hole contour in red
            cv2.drawContours(vis3, [shaft_contour], -1, (0, 0, 255), 2)

            # Draw fitted ellipse in yellow
            cv2.ellipse(vis3, ellipse, (0, 255, 255), 2)

            # Draw gear centre crosshair in green
            cv2.line(vis3, (gear_cx - 40, gear_cy),
                     (gear_cx + 40, gear_cy), (0, 255, 0), 1)
            cv2.line(vis3, (gear_cx, gear_cy - 40),
                     (gear_cx, gear_cy + 40), (0, 255, 0), 1)
            cv2.circle(vis3, (gear_cx, gear_cy), 4, (0, 255, 0), -1)

            # Label
            cv2.putText(vis3,
                        f"Centre ({gear_cx},{gear_cy})",
                        (gear_cx + 8, gear_cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(vis3,
                        f"Shaft d={shaft_diameter_px:.1f}px",
                        (gear_cx + 8, gear_cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            axes[2].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Result\n"
                               f"red=contour  yellow=fitted ellipse  "
                               f"green=centre")
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig("results/find_centre.png", dpi=150)
            plt.show()
            print("\nSaved to results/find_centre.png")