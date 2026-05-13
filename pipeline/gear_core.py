import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def fit_circle_to_points(points):
    """
    Fits a circle to a set of (x,y) points using algebraic least squares.
    Returns (cx, cy, radius).
    """
    pts    = np.array(points, dtype=np.float64)
    xi, yi = pts[:, 0], pts[:, 1]
    A      = np.column_stack([2*xi, 2*yi, np.ones(len(xi))])
    b_vec  = xi**2 + yi**2
    result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    ecx, ecy, c     = result
    radius           = np.sqrt(c + ecx**2 + ecy**2)
    return float(ecx), float(ecy), float(radius)


def find_gear_centre(gray, search_fraction=0.85,
                     fill_tolerance=15):
    """
    Finds gear centre using a two-stage approach:

    Stage 1: Find gear boundary from background contrast
             The gear is bright against dark background.
             Threshold → find largest circular blob → 
             that blob's centroid = gear centre estimate.

    Stage 2: Find shaft hole near that centre
             Search only within inner 20% of gear radius
             for the largest circular dark region.
             This avoids auxiliary holes at 120° intervals
             which are smaller than the shaft hole.
    """
    h, w = gray.shape

    # ══════════════════════════════════════════════════════
    # STAGE 1 — Find gear body and centre from outer boundary
    # ══════════════════════════════════════════════════════

    blurred = cv2.GaussianBlur(gray, (15, 15), sigmaX=0)

    # Otsu threshold — separates gear (bright) from background (dark)
    otsu_val, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"  Otsu threshold  : {otsu_val:.0f}")

    # Find connected components
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Pick largest non-background component — that's the gear body
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        # Try inverted
        binary     = cv2.bitwise_not(binary)
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]

    gear_label    = int(np.argmax(areas)) + 1
    gear_area     = int(stats[gear_label, cv2.CC_STAT_AREA])
    gear_cx_rough = float(centroids[gear_label][0])
    gear_cy_rough = float(centroids[gear_label][1])

    print(f"  Gear body area  : {gear_area} px")
    print(f"  Gear centre est : ({gear_cx_rough:.0f}, {gear_cy_rough:.0f})")

    # Estimate gear radius from area
    gear_r_est = float(np.sqrt(gear_area / np.pi))
    print(f"  Gear radius est : {gear_r_est:.1f} px")

    # Refine centre using contour of gear body
    gear_mask  = np.uint8(labels == gear_label) * 255
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    gear_mask  = cv2.morphologyEx(
        gear_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        gear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        gear_contour = max(contours, key=cv2.contourArea)
        if len(gear_contour) >= 5:
            ellipse      = cv2.fitEllipse(gear_contour)
            gear_cx_rough = float(ellipse[0][0])
            gear_cy_rough = float(ellipse[0][1])
            gear_r_est    = float(
                (ellipse[1][0] + ellipse[1][1]) / 4)
            print(f"  Refined centre  : "
                  f"({gear_cx_rough:.0f}, {gear_cy_rough:.0f})")
            print(f"  Refined radius  : {gear_r_est:.1f} px")

    gear_cx_rough = int(np.clip(gear_cx_rough, 0, w-1))
    gear_cy_rough = int(np.clip(gear_cy_rough, 0, h-1))

    # ══════════════════════════════════════════════════════
    # STAGE 2 — Find shaft hole near gear centre
    # ══════════════════════════════════════════════════════
    # Search only within inner 25% of gear radius
    # Shaft hole is always at/near gear rotational centre
    # Auxiliary holes (sensor plate) are at ~70-80% radius

    shaft_search_r = int(gear_r_est * 0.25)
    shaft_search_r = max(shaft_search_r, 20)

    sx1 = max(0, gear_cx_rough - shaft_search_r)
    sx2 = min(w, gear_cx_rough + shaft_search_r)
    sy1 = max(0, gear_cy_rough - shaft_search_r)
    sy2 = min(h, gear_cy_rough + shaft_search_r)

    shaft_roi = blurred[sy1:sy2, sx1:sx2]

    print(f"  Shaft search r  : {shaft_search_r} px")
    print(f"  Shaft ROI size  : "
          f"{shaft_roi.shape[1]}x{shaft_roi.shape[0]}")

    # Threshold shaft ROI — shaft hole is dark
    shaft_otsu, shaft_bin = cv2.threshold(
        shaft_roi, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print(f"  Shaft Otsu      : {shaft_otsu:.0f}")

    # Find dark connected components in shaft ROI
    n_lab, s_labels, s_stats, s_centroids = \
        cv2.connectedComponentsWithStats(shaft_bin, connectivity=8)

    best_label = -1
    best_score = -1

    for lbl in range(1, n_lab):
        area = s_stats[lbl, cv2.CC_STAT_AREA]
        if area < 30:
            continue

        # Get contour for circularity
        comp_mask  = np.uint8(s_labels == lbl) * 255
        c_contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c_contours:
            continue

        cnt       = c_contours[0]
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / perimeter**2
        if circularity < 0.4:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            best_label = lbl

    if best_label == -1:
        # No shaft hole found — use gear centre as fallback
        print("  WARNING: Shaft hole not found — using gear centre")
        gear_cx      = gear_cx_rough
        gear_cy      = gear_cy_rough
        shaft_radius = gear_r_est * 0.10

        # Create dummy contour and ellipse
        pts = []
        for a in np.linspace(0, 2*np.pi, 20, endpoint=False):
            pts.append([[int(gear_cx + shaft_radius*np.cos(a)),
                         int(gear_cy + shaft_radius*np.sin(a))]])
        shaft_contour = np.array(pts, dtype=np.int32)
        shaft_ellipse = (
            (float(gear_cx), float(gear_cy)),
            (float(shaft_radius*2), float(shaft_radius*2)),
            0.0)

    else:
        # Extract shaft hole details
        shaft_mask_full = np.zeros((h, w), dtype=np.uint8)
        shaft_region    = np.uint8(s_labels == best_label) * 255

        # Map back to full image coordinates
        shaft_mask_full[sy1:sy2, sx1:sx2] = shaft_region

        s_contours, _ = cv2.findContours(
            shaft_mask_full, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        if not s_contours:
            shaft_contour = np.array(
                [[[gear_cx_rough, gear_cy_rough]]], dtype=np.int32)
            shaft_ellipse = (
                (float(gear_cx_rough), float(gear_cy_rough)),
                (20.0, 20.0), 0.0)
            shaft_radius  = 10.0
            gear_cx       = gear_cx_rough
            gear_cy       = gear_cy_rough
        else:
            shaft_contour = max(s_contours, key=cv2.contourArea)

            if len(shaft_contour) >= 5:
                shaft_ellipse = cv2.fitEllipse(shaft_contour)
                gear_cx       = int(shaft_ellipse[0][0])
                gear_cy       = int(shaft_ellipse[0][1])
                shaft_radius  = float(
                    (shaft_ellipse[1][0] +
                     shaft_ellipse[1][1]) / 4)
            else:
                M = cv2.moments(shaft_contour)
                if M['m00'] > 0:
                    gear_cx = int(M['m10'] / M['m00'])
                    gear_cy = int(M['m01'] / M['m00'])
                else:
                    gear_cx = gear_cx_rough
                    gear_cy = gear_cy_rough
                shaft_radius  = float(
                    np.sqrt(s_stats[best_label,
                                    cv2.CC_STAT_AREA] / np.pi))
                shaft_ellipse = (
                    (float(gear_cx), float(gear_cy)),
                    (shaft_radius*2, shaft_radius*2), 0.0)

    print(f"  Shaft centre    : ({gear_cx}, {gear_cy})")
    print(f"  Shaft radius    : {shaft_radius:.1f} px")

    return (int(gear_cx), int(gear_cy),
            shaft_contour, shaft_ellipse, float(shaft_radius))


def find_tip_circle(gray, gear_cx, gear_cy,
                    shaft_radius_px, darkness_margin=15):
    """
    Finds tip circle by casting rays and finding the outermost
    bright region before the background.

    Handles gears with internal step cuts by ignoring intermediate
    dark rings and finding the true outer boundary.
    """
    shaft_radius_px = float(shaft_radius_px)
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=0)

    # ── Estimate background brightness from image corners ─────────────────
    corner_samples = [
        blurred[10,    10   ],
        blurred[10,    w-10 ],
        blurred[h-10,  10   ],
        blurred[h-10,  w-10 ],
        blurred[h//2,  10   ],
        blurred[h//2,  w-10 ],
        blurred[10,    w//2 ],
        blurred[h-10,  w//2 ],
    ]
    bg_brightness = float(np.median(corner_samples))

    # ── Estimate gear body brightness ─────────────────────────────────────
    sample_angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    body_r        = int(shaft_radius_px * 2.5)
    body_r        = min(body_r, int(min(h, w) * 0.2))
    body_samples  = []
    for a in sample_angles:
        sx = np.clip(int(gear_cx + body_r*np.cos(a)), 0, w-1)
        sy = np.clip(int(gear_cy + body_r*np.sin(a)), 0, h-1)
        body_samples.append(float(blurred[sy, sx]))
    gear_brightness = float(np.median(body_samples))

    # ── Background threshold ───────────────────────────────────────────────
    # Pixel is background if below this value
    # Set at 30% of the way from background to gear body
    bg_threshold = bg_brightness + (
        gear_brightness - bg_brightness) * 0.60
    bg_threshold = max(bg_threshold, bg_brightness + 20)

    print(f"  Background      : {bg_brightness:.1f}")
    print(f"  Gear body       : {gear_brightness:.1f}")
    print(f"  Tip threshold   : {bg_threshold:.1f}")

    # ── Cast 360 rays ──────────────────────────────────────────────────────
    NUM_ANGLES  = 360
    angles      = np.linspace(0, 2*np.pi, NUM_ANGLES, endpoint=False)
    tip_radii   = []
    edge_points = []

    # Max ray length — stop 3px before image edge
    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # How far can this ray go before hitting image edge
        if abs(cos_a) > 1e-6:
            rx = (w-3 - gear_cx) / cos_a if cos_a > 0 \
                 else (3 - gear_cx) / cos_a
        else:
            rx = float(min(h, w))

        if abs(sin_a) > 1e-6:
            ry = (h-3 - gear_cy) / sin_a if sin_a > 0 \
                 else (3 - gear_cy) / sin_a
        else:
            ry = float(min(h, w))

        max_r = int(max(0, min(rx, ry)))

        if max_r < shaft_radius_px + 5:
            continue

        # Sample full ray
        start_r = int(shaft_radius_px + 5)
        radii   = np.arange(start_r, max_r)
        if len(radii) == 0:
            continue

        xs   = np.clip(
            (gear_cx + radii * cos_a).astype(int), 0, w-1)
        ys   = np.clip(
            (gear_cy + radii * sin_a).astype(int), 0, h-1)
        vals = blurred[ys, xs].astype(float)

        # Find outermost pixel above background threshold
        above_bg = vals > bg_threshold
        if not np.any(above_bg):
            continue

        last_idx = int(np.where(above_bg)[0][-1])
        tip_r    = float(radii[last_idx])
        tip_x    = int(gear_cx + tip_r * cos_a)
        tip_y    = int(gear_cy + tip_r * sin_a)

        tip_radii.append(tip_r)
        edge_points.append((tip_x, tip_y))

    if len(tip_radii) < 10:
        raise RuntimeError(
            f"Tip detection failed — only {len(tip_radii)} rays "
            f"found edge. Check background contrast.")

    tip_radii = np.array(tip_radii, dtype=float)

    # ── Remove outliers ────────────────────────────────────────────────────
    # Rays that hit image boundary give wrong values
    median_r = float(np.median(tip_radii))
    std_r    = float(np.std(tip_radii))
    valid    = (tip_radii > median_r - 2*std_r) & \
               (tip_radii < median_r + 2*std_r)

    tip_radii_clean  = tip_radii[valid]
    edge_points_arr  = np.array(edge_points, dtype=float)
    edge_points_clean = edge_points_arr[valid]

    print(f"  Tip median r    : {median_r:.1f} px")
    print(f"  Tip std         : {std_r:.1f} px")
    print(f"  Valid rays      : {np.sum(valid)}/{NUM_ANGLES}")

    # ── Constrained circle fit ─────────────────────────────────────────────
    # Use gear centre as fixed centre — more robust than free fit
    # when some rays are clipped at image boundary
    dists = np.sqrt(
        (edge_points_clean[:, 0] - gear_cx)**2 +
        (edge_points_clean[:, 1] - gear_cy)**2)

    # Second outlier pass — remove rays significantly off median
    med2  = float(np.median(dists))
    std2  = float(np.std(dists))
    valid2 = (dists > med2 - 1.5*std2) & \
              (dists < med2 + 1.5*std2)

    tip_radius = float(np.mean(dists[valid2]))

    print(f"  Tip radius      : {tip_radius:.1f} px")

    # Regenerate clean edge points on fitted circle
    clean_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    tip_xs = gear_cx + tip_radius * np.cos(clean_angles)
    tip_ys = gear_cy + tip_radius * np.sin(clean_angles)
    edge_points_final = list(zip(
        tip_xs.astype(int).tolist(),
        tip_ys.astype(int).tolist()))

    return tip_radius, tip_radius * 2, edge_points_final


def find_true_pitch_radius(gray, gear_cx, gear_cy,
                           root_radius_px, tip_radius_px,
                           num_angles=1440):
    """
    Finds the true pitch radius as the radius where tooth arc
    equals gap arc (50% duty cycle).

    Scans from root to tip, at each radius computing what fraction
    of the circumference is tooth (bright) vs gap (dark).
    The pitch radius is where that fraction crosses 50%.

    Returns:
        pitch_radius_px  - true pitch radius
        duty_cycle_curve - (radii, duty_cycles) for plotting
    """
    h, w      = gray.shape
    angles    = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    cos_a     = np.cos(angles)
    sin_a     = np.sin(angles)

    scan_radii   = np.arange(int(root_radius_px),
                              int(tip_radius_px) + 1)
    duty_cycles  = []

    for r in scan_radii:
        xs   = np.clip((gear_cx + r * cos_a).astype(int), 0, w-1)
        ys   = np.clip((gear_cy + r * sin_a).astype(int), 0, h-1)
        vals = gray[ys, xs].astype(float)

        # Adaptive threshold for this radius
        thresh      = vals.min() + (vals.max() - vals.min()) * 0.5
        tooth_frac  = float(np.mean(vals > thresh))
        duty_cycles.append(tooth_frac)

    duty_cycles = np.array(duty_cycles)

    # Find where duty cycle crosses 0.5 (tooth = gap)
    # Search from the tip inward
    pitch_radius_px = float(tip_radius_px)   # default fallback

    for i in range(len(duty_cycles) - 1, 0, -1):
        if duty_cycles[i] >= 0.5 and duty_cycles[i-1] < 0.5:
            # Interpolate crossing point
            frac = (0.5 - duty_cycles[i-1]) / (duty_cycles[i] - duty_cycles[i-1])
            pitch_radius_px = float(scan_radii[i-1] + frac)
            break
        elif duty_cycles[i] <= 0.5 and duty_cycles[i-1] > 0.5:
            frac = (0.5 - duty_cycles[i]) / (duty_cycles[i-1] - duty_cycles[i])
            pitch_radius_px = float(scan_radii[i] - frac)
            break

    return pitch_radius_px, (scan_radii, duty_cycles)


def find_pitch_and_root(gray, gear_cx, gear_cy,
                        shaft_radius_px, tip_radius_px,
                        num_angles=720):
    """
    Finds root circle and computes pitch circle from radial profile.

    The root circle is detected as the brightness valley in the
    radial profile just inside the tip circle. Pitch circle is
    computed from tip radius and addendum (tooth_depth / 2.25).

    Parameters:
        gray             - grayscale image
        gear_cx, gear_cy - gear centre
        shaft_radius_px  - shaft hole radius
        tip_radius_px    - tip circle radius

    Returns dict with:
        root_radius_px, root_diameter_px
        pitch_radius_px, pitch_diameter_px
        tooth_depth_px, addendum_px, dedendum_px
    """
    h, w = gray.shape

    scan_start = int(shaft_radius_px) + 5
    scan_end   = int(tip_radius_px)  + 15

    angles    = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    cos_a     = np.cos(angles)
    sin_a     = np.sin(angles)

    radii        = np.arange(scan_start, scan_end)
    radial_mean  = []

    for r in radii:
        xs   = np.clip((gear_cx + r * cos_a).astype(int), 0, w-1)
        ys   = np.clip((gear_cy + r * sin_a).astype(int), 0, h-1)
        radial_mean.append(gray[ys, xs].mean())

    radial_mean = np.array(radial_mean)

    # Focus on gear body only (inside tip circle)
    body_mask   = radii <= tip_radius_px
    body_radii  = radii[body_mask]
    body_mean   = radial_mean[body_mask]

    # Smooth to remove tooth-to-tooth variation
    win = min(11, len(body_mean) - 1 if len(body_mean) % 2 == 0
              else len(body_mean) - 2)
    win = max(win, 5)
    # Guard: window must be odd, >= 3, < signal length
    n_sig = len(body_mean)
    win   = min(win, n_sig - 1)
    win   = max(win, 3)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)

    if n_sig < 4:
        smoothed = body_mean.copy()
    else:
        smoothed = savgol_filter(
            body_mean, window_length=win,
            polyorder=min(3, win-1))
    # Find valleys — root circle is outermost valley
    valleys, _ = find_peaks(-smoothed, prominence=5, distance=10)

    if len(valleys) == 0:
        raise RuntimeError("No root circle valley found in radial profile")

    root_idx       = valleys[np.argmax(body_radii[valleys])]
    root_radius    = float(body_radii[root_idx])
    tooth_depth    = tip_radius_px - root_radius
    addendum       = tooth_depth / 2.25
    dedendum       = tooth_depth - addendum
    pitch_radius   = tip_radius_px - addendum

    return {
        'root_radius_px'    : root_radius,
        'root_diameter_px'  : root_radius * 2,
        'pitch_radius_px'   : pitch_radius,
        'pitch_diameter_px' : pitch_radius * 2,
        'tooth_depth_px'    : tooth_depth,
        'addendum_px'       : addendum,
        'dedendum_px'       : dedendum,
    }


def run_core_pipeline(image_path, px_per_mm=1.0,
                      search_fraction=0.30,
                      fill_tolerance=30,
                      darkness_margin=15):
    """
    Runs the full core pipeline on one image.

    Parameters:
        image_path      - path to gear image
        px_per_mm       - calibration factor (pixels per mm)
        search_fraction - fraction of image to search for shaft
        fill_tolerance  - brightness tolerance for shaft detection
        darkness_margin - darkness threshold for tip circle detection

    Returns:
        measurements - dict of all measurements in px and mm
        intermediates - dict of intermediate results for visualisation
    """
    img  = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    # Resize large images to max 1500px wide for consistent processing
    h_orig, w_orig = img.shape[:2]
    max_dim = 1500
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img   = cv2.resize(img, (new_w, new_h),
                           interpolation=cv2.INTER_AREA)
        print(f"  Resized: {w_orig}x{h_orig} → {new_w}x{new_h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Find gear centre via shaft hole ────────────────────────────
    gear_cx, gear_cy, shaft_contour, shaft_ellipse, shaft_radius_px = \
        find_gear_centre(gray, search_fraction)

    
    # ── Step 2: Find tip circle ────────────────────────────────────────────
    tip_radius_px, tip_diam_px, edge_points = \
        find_tip_circle(gray, gear_cx, gear_cy,
                        shaft_radius_px, darkness_margin)
    
    tip_radius_px = float(tip_radius_px)
    tip_diam_px   = tip_radius_px * 2

    # ── Step 3: Find root and pitch circles ───────────────────────────────
    # ── Step 3a: Find root circle ───────────────────────────────────────────
    circles = find_pitch_and_root(
        gray, gear_cx, gear_cy,
        shaft_radius_px, tip_radius_px)
    
    # ── Step 3b: Find true pitch radius from duty cycle ────────────────────
    true_pitch_r, duty_data = find_true_pitch_radius(
        gray, gear_cx, gear_cy,
        circles['root_radius_px'], tip_radius_px)

    # Override computed pitch with measured pitch
    circles['pitch_radius_px']  = true_pitch_r
    circles['pitch_diameter_px'] = true_pitch_r * 2
   
    # ── Step 4: Circularity ────────────────────────────────────────────────
    circ_results = compute_circularity(
        edge_points, gear_cx, gear_cy, tip_radius_px)
        
    # ── Assemble measurements ──────────────────────────────────────────────
    def px2mm(px): return px / px_per_mm

    measurements = {
        # In pixels
        'circularity'           : circ_results['circularity_iso'],
        'circularity_gdt_zone'  : circ_results['circularity_gdt_zone'],
        'circularity_gdt_std'   : circ_results['circularity_gdt_std'],
        'gear_centre_x'         : gear_cx,
        'gear_centre_y'         : gear_cy,
        'shaft_diameter_px'     : shaft_radius_px*2,
        'tip_diameter_px'       : tip_diam_px,
        'root_diameter_px'      : circles['root_diameter_px'],
        'pitch_diameter_px'     : circles['pitch_diameter_px'],
        'tooth_depth_px'        : circles['tooth_depth_px'],
        'addendum_px'           : circles['addendum_px'],
        'dedendum_px'           : circles['dedendum_px'],


        # In mm
        'shaft_diameter_mm'     : px2mm(shaft_radius_px*2),
        'tip_diameter_mm'       : px2mm(tip_diam_px),
        'root_diameter_mm'      : px2mm(circles['root_diameter_px']),
        'pitch_diameter_mm'     : px2mm(circles['pitch_diameter_px']),
        'tooth_depth_mm'        : px2mm(circles['tooth_depth_px']),

        # Scale
        'px_per_mm'             : px_per_mm,
    }

    intermediates = {
        'gray'           : gray,
        'img'            : img,
        'shaft_contour'  : shaft_contour,
        'shaft_ellipse'  : shaft_ellipse,
        'edge_points'    : edge_points,
        'tip_radius_px'  : tip_radius_px,
        'root_radius_px' : circles['root_radius_px'],
        'pitch_radius_px': circles['pitch_radius_px'],
        'shaft_radius_px': shaft_radius_px,
        'duty_data'      : duty_data,
    }

    return measurements, intermediates


def compute_circularity(edge_points, gear_cx, gear_cy,
                        tip_radius_px):
    """
    Computes two circularity measures:

    1. ISO circularity: 4π × area / perimeter²
       Range 0-1. Easy to compute, good for comparison.
       Not GD&T compliant.

    2. GD&T roundness (approximation):
       Standard deviation of radial distances from fitted centre.
       Lower = more circular. Units = pixels (convertible to mm).
       This approximates the GD&T tolerance zone width.

    Parameters:
        edge_points    - list of (x,y) points on gear boundary
        gear_cx, gcy   - gear centre
        tip_radius_px  - fitted tip circle radius

    Returns dict with both measures.
    """
    pts  = np.array(edge_points, dtype=np.float64)

    # ── ISO circularity ────────────────────────────────────────────────────
    contour    = pts.reshape(-1, 1, 2).astype(np.int32)
    area       = cv2.contourArea(contour)
    perimeter  = cv2.arcLength(contour, closed=True)
    iso_circ   = float(4 * np.pi * area / perimeter**2) \
                 if perimeter > 0 else 0.0

    # ── GD&T roundness approximation ──────────────────────────────────────
    # Radial distance of each point from fitted tip circle centre
    dx         = pts[:, 0] - gear_cx
    dy         = pts[:, 1] - gear_cy
    radii      = np.sqrt(dx**2 + dy**2)

    # Circumscribed circle = max radius
    # Inscribed circle     = min radius
    # GD&T tolerance zone  = difference between them
    r_max      = float(radii.max())
    r_min      = float(radii.min())
    r_std      = float(radii.std())

    gdt_zone   = r_max - r_min          # tolerance zone width in px
    gdt_approx = r_std * 2              # ≈ tolerance zone (normal assumption)

    return {
        'circularity_iso'      : iso_circ,
        'circularity_gdt_zone' : gdt_zone,
        'circularity_gdt_std'  : r_std,
        'radius_max_px'        : r_max,
        'radius_min_px'        : r_min,
    }