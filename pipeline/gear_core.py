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


def find_gear_centre(gray, search_fraction=0.30, fill_tolerance=30):
    """
    Finds gear centre by locating the shaft hole.

    Strategy:
        1. Find darkest pixel near image centre (shaft hole)
        2. Threshold + connected component to isolate shaft hole
        3. Fit ellipse to shaft hole contour → gear centre

    Parameters:
        gray            - grayscale image
        search_fraction - fraction of image to search around centre
        fill_tolerance  - brightness range above darkest pixel to include

    Returns:
        gear_cx, gear_cy     - gear centre in pixels
        shaft_diameter_px    - shaft hole diameter
        shaft_contour        - contour points of shaft hole
        shaft_ellipse        - fitted ellipse parameters
    """
    h, w = gray.shape
    img_cx, img_cy = w // 2, h // 2

    # ── Find darkest pixel near image centre ───────────────────────────────
    search_radius = int(min(w, h) * search_fraction)
    search_mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(search_mask, (img_cx, img_cy), search_radius, 255, -1)

    masked = gray.copy()
    masked[search_mask == 0] = 255

    min_val, _, min_loc, _ = cv2.minMaxLoc(masked)
    seed_x, seed_y = min_loc

    # ── Threshold and connected component ─────────────────────────────────
    shaft_threshold = int(min_val) + fill_tolerance
    _, shaft_mask   = cv2.threshold(
        gray, shaft_threshold, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        shaft_mask, connectivity=8)

    seed_label = labels[seed_y, seed_x]
    if seed_label == 0:
        raise RuntimeError(
            "Shaft seed pixel in background — increase fill_tolerance")

    # ── Isolate shaft hole component ───────────────────────────────────────
    component = np.zeros((h, w), dtype=np.uint8)
    component[labels == seed_label] = 255

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Fit ellipse to shaft hole ──────────────────────────────────────────
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise RuntimeError("No shaft hole contour found")

    shaft_contour = max(contours, key=cv2.contourArea)

    if len(shaft_contour) < 5:
        raise RuntimeError("Shaft contour too small for ellipse fit")

    ellipse           = cv2.fitEllipse(shaft_contour)
    (cx, cy), (a, b), angle = ellipse

    gear_cx           = int(round(cx))
    gear_cy           = int(round(cy))
    shaft_diameter_px = (a + b) / 2

    return gear_cx, gear_cy, shaft_diameter_px, shaft_contour, ellipse


def find_tip_circle(gray, gear_cx, gear_cy, shaft_radius_px,
                    darkness_margin=15, max_radius_fraction=0.55,
                    num_angles=360):
    """
    Finds gear tip circle (outer diameter) by radial ray casting.

    Casts rays outward from gear centre. Each ray finds the first
    dark pixel beyond the shaft hole — that's the outer dark ring
    which marks the gear tip circle boundary.

    Parameters:
        gray                 - grayscale image
        gear_cx, gear_cy     - gear centre
        shaft_radius_px      - shaft hole radius (scan starts here)
        darkness_margin      - how dark a pixel must be to count as edge
        max_radius_fraction  - maximum search radius as fraction of image
        num_angles           - number of rays to cast

    Returns:
        tip_radius_px    - fitted tip circle radius
        tip_diameter_px  - fitted tip circle diameter
        edge_points      - list of (x,y) edge points found
        fit_centre       - (cx, cy) of fitted circle
    """
    h, w = gray.shape

    # Get minimum dark value (from shaft hole, already known)
    search_r      = int(shaft_radius_px)
    region        = gray[
        max(0, gear_cy - search_r):gear_cy + search_r,
        max(0, gear_cx - search_r):gear_cx + search_r
    ]
    min_val       = int(region.min()) if region.size > 0 else 20
    edge_threshold = min_val + darkness_margin

    scan_start    = int(shaft_radius_px) + 10
    max_radius    = int(min(w, h) * max_radius_fraction)

    angles        = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    edge_points   = []
    edge_radii    = []

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for r in range(scan_start, max_radius):
            x = int(round(gear_cx + r * cos_a))
            y = int(round(gear_cy + r * sin_a))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if gray[y, x] <= edge_threshold:
                edge_points.append((x, y))
                edge_radii.append(r)
                break

    if len(edge_points) < 10:
        raise RuntimeError(
            f"Too few edge points ({len(edge_points)}) — "
            f"adjust darkness_margin")

    # ── Fit circle to edge points ──────────────────────────────────────────
    def fit_circle(points):
        pts    = np.array(points, dtype=np.float64)
        xi, yi = pts[:, 0], pts[:, 1]
        A      = np.column_stack([2*xi, 2*yi, np.ones(len(xi))])
        b_vec  = xi**2 + yi**2
        result, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        ecx, ecy, c = result
        r = np.sqrt(c + ecx**2 + ecy**2)
        return float(ecx), float(ecy), float(r)

    fcx, fcy, tip_radius = fit_circle(edge_points)

    return tip_radius, tip_radius * 2, edge_points, (fcx, fcy)


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
    smoothed = savgol_filter(body_mean, window_length=win, polyorder=3)

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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Find gear centre via shaft hole ────────────────────────────
    gear_cx, gear_cy, shaft_diam_px, shaft_contour, shaft_ellipse = \
        find_gear_centre(gray, search_fraction, fill_tolerance)

    shaft_radius_px = shaft_diam_px / 2

    # ── Step 2: Find tip circle ────────────────────────────────────────────
    tip_radius_px, tip_diam_px, edge_points, fit_centre = \
        find_tip_circle(gray, gear_cx, gear_cy,
                        shaft_radius_px, darkness_margin)

    # ── Step 3: Find root and pitch circles ───────────────────────────────
    # ── Step 3: Find root circle ───────────────────────────────────────────
    circles = find_pitch_and_root(
        gray, gear_cx, gear_cy, shaft_radius_px, tip_radius_px)

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
        'shaft_diameter_px'     : shaft_diam_px,
        'tip_diameter_px'       : tip_diam_px,
        'root_diameter_px'      : circles['root_diameter_px'],
        'pitch_diameter_px'     : circles['pitch_diameter_px'],
        'tooth_depth_px'        : circles['tooth_depth_px'],
        'addendum_px'           : circles['addendum_px'],
        'dedendum_px'           : circles['dedendum_px'],


        # In mm
        'shaft_diameter_mm'     : px2mm(shaft_diam_px),
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
        'duty_data'      : duty_data,       # ← add this
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