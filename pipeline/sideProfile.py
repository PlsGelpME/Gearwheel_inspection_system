import cv2
import numpy as np
from scipy.signal import savgol_filter


def find_face_edges(gray, roi_bottom_fraction=0.95):
    """
    Finds top and bottom gear face edges adaptively.
    Handles varying image sizes and gear positions.
    """
    h, w = gray.shape

    # ── Resize if image is very large ─────────────────────────────────────
    # Standardise to max 1000px wide for consistent parameter behaviour
    scale  = 1.0
    if w > 1000:
        scale    = 1000.0 / w
        new_w    = 1000
        new_h    = int(h * scale)
        gray_use = cv2.resize(gray, (new_w, new_h))
    else:
        gray_use = gray
        new_h, new_w = h, w

    # ── Preprocessing ──────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray_use, (5, 5), sigmaX=0)

    # Use Otsu threshold to handle varying brightness
    otsu_val, _ = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny with auto thresholds based on Otsu value
    t1 = int(otsu_val * 0.5)
    t2 = int(otsu_val * 1.0)
    t1 = max(t1, 10)
    t2 = max(t2, 30)

    edges = cv2.Canny(blurred, t1, t2)

    # Mask bottom 20% — base/background noise
    edges[int(new_h * roi_bottom_fraction):, :] = 0

    # Also mask top 5% — unlikely to have gear face there
    edges[:int(new_h * 0.05), :] = 0

    print(f"  Otsu threshold  : {otsu_val:.0f}  "
          f"Canny: ({t1}, {t2})")
    print(f"  Working size    : {new_w} x {new_h}  "
          f"(scale={scale:.3f})")

    # ── Hough lines — adaptive minLineLength ──────────────────────────────
    # Use 15% of image width as minimum — shorter than before
    min_len = max(30, int(new_w * 0.12))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,          # lower threshold
        minLineLength=min_len,
        maxLineGap=40          # larger gap bridging
    )

    if lines is None:
        raise RuntimeError("No Hough lines found")

    print(f"  Hough lines     : {len(lines)}  "
          f"(minLen={min_len})")

    # ── Filter and categorise ──────────────────────────────────────────────
    horizontal = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle  = np.degrees(np.arctan2(y2-y1, x2-x1))
        y_mid  = (y1 + y2) / 2
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Horizontal within ±15°
        if abs(angle) <= 15 or abs(angle) >= 165:
            horizontal.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'angle' : float(angle),
                'y_mid' : float(y_mid),
                'length': float(length),
            })

    print(f"  Horizontal lines: {len(horizontal)}")
    for l in sorted(horizontal, key=lambda x: x['y_mid']):
        print(f"    y={l['y_mid']:6.1f}  "
              f"len={l['length']:5.0f}  "
              f"angle={l['angle']:5.1f}°")

    if len(horizontal) < 2:
        # Fallback — use column scanning
        print("  Falling back to column scan...")
        return _column_scan_edges(gray_use, edges, new_w, new_h, scale)

    # ── Cluster by y ───────────────────────────────────────────────────────
    y_vals     = np.array([l['y_mid'] for l in horizontal])
    sorted_idx = np.argsort(y_vals)
    sorted_ys  = y_vals[sorted_idx]
    sorted_h   = [horizontal[i] for i in sorted_idx]

    gaps     = np.diff(sorted_ys)
    clusters = []
    current  = [sorted_h[0]]

    for i, gap in enumerate(gaps):
        if gap > 15:
            clusters.append(current)
            current = []
        current.append(sorted_h[i+1])
    clusters.append(current)

    print(f"  Clusters        : {len(clusters)}")

    if len(clusters) < 2:
        print("  Only 1 cluster — using column scan fallback")
        return _column_scan_edges(gray_use, edges, new_w, new_h, scale)

    # Sort clusters by y — top cluster = top edge, bottom = bottom edge
    clusters.sort(key=lambda c: np.mean([l['y_mid'] for l in c]))

    # Pick the two clusters with longest total line length
    # (gear face edges are longer than random noise lines)
    def cluster_length(c):
        return sum(l['length'] for l in c)

    # Among all clusters, pick two with highest combined y separation
    # and highest line length — these are the gear face edges
    if len(clusters) == 2:
        top_cluster = clusters[0]
        bot_cluster = clusters[1]
    else:
        # More than 2 clusters
        # Pick the pair that gives the most plausible face width
        # Face width should be between 5% and 40% of image height
        min_fw = new_h * 0.05
        max_fw = new_h * 0.40

        best_pair  = None
        best_score = -1

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                y_i = np.mean([l['y_mid'] for l in clusters[i]])
                y_j = np.mean([l['y_mid'] for l in clusters[j]])
                gap = y_j - y_i

                if not (min_fw <= gap <= max_fw):
                    continue

                # Score = total line length of both clusters
                score = (sum(l['length'] for l in clusters[i]) +
                         sum(l['length'] for l in clusters[j]))

                if score > best_score:
                    best_score = score
                    best_pair  = (clusters[i], clusters[j])

        if best_pair is None:
            # Fallback — just use first and last cluster
            best_pair = (clusters[0], clusters[-1])

        top_cluster, bot_cluster = best_pair

    def fit_cluster(cluster):
        xs = np.array([l['x1'] for l in cluster] +
                      [l['x2'] for l in cluster], dtype=float)
        ys = np.array([l['y1'] for l in cluster] +
                      [l['y2'] for l in cluster], dtype=float)
        m, c  = np.polyfit(xs, ys, 1)
        angle = float(np.degrees(np.arctan(m)))
        return float(m), float(c), angle

    m_top, c_top, a_top = fit_cluster(top_cluster)
    m_bot, c_bot, a_bot = fit_cluster(bot_cluster)

    # ── Scale back to original image coordinates ───────────────────────────
    # Line equation in scaled image: y = m*x + c
    # In original image: y_orig = y_scaled / scale
    #                    x_orig = x_scaled / scale
    # So: y_orig/scale = m * (x_orig/scale) + c
    #     y_orig = m * x_orig + c * scale
    c_top_orig = c_top / scale
    c_bot_orig = c_bot / scale

    top_y = float(m_top * (w/2) + c_top_orig)
    bot_y = float(m_bot * (w/2) + c_bot_orig)

    m_avg      = (m_top + m_bot) / 2
    face_width = abs(c_bot_orig - c_top_orig) / np.sqrt(m_avg**2 + 1)

    # x extent
    def xspan(cluster):
        xs = ([l['x1'] for l in cluster] +
              [l['x2'] for l in cluster])
        return int(min(xs)/scale), int(max(xs)/scale)

    x_start, x_end = xspan(top_cluster)

    print(f"  Top edge y      : {top_y:.1f}  angle={a_top:.2f}°")
    print(f"  Bottom edge y   : {bot_y:.1f}  angle={a_bot:.2f}°")
    print(f"  Face width      : {face_width:.1f} px")

    return {
        'top_y'         : top_y,
        'bot_y'         : bot_y,
        'm_top'         : m_top,
        'c_top'         : c_top_orig,
        'm_bot'         : m_bot,
        'c_bot'         : c_bot_orig,
        'angle_top_deg' : a_top,
        'angle_bot_deg' : a_bot,
        'face_angle_deg': (a_top + a_bot) / 2,
        'face_width_px' : float(face_width),
        'x_start'       : x_start,
        'x_end'         : x_end,
        'scale'         : scale,
    }


def _column_scan_edges(gray, edges, w, h, scale):
    """
    Fallback edge detection using column scanning.
    Used when Hough line detection doesn't find enough horizontal lines.
    """
    x_start = int(w * 0.05)
    x_end   = int(w * 0.95)

    all_xs, all_tops, all_bots = [], [], []

    for x in range(x_start, x_end):
        col       = edges[:, x]
        edge_rows = np.where(col > 0)[0]
        if len(edge_rows) >= 2:
            all_xs.append(x)
            all_tops.append(int(edge_rows[0]))
            all_bots.append(int(edge_rows[-1]))

    if len(all_xs) < 10:
        raise RuntimeError("Column scan also failed — "
                           "check image quality")

    all_xs   = np.array(all_xs,   dtype=float)
    all_tops = np.array(all_tops, dtype=float)
    all_bots = np.array(all_bots, dtype=float)

    top_med  = np.median(all_tops)
    bot_med  = np.median(all_bots)
    top_mask = np.abs(all_tops - top_med) < 40
    bot_mask = np.abs(all_bots - bot_med) < 50

    def fit(xs, ys):
        m, c  = np.polyfit(xs, ys, 1)
        return float(m), float(c), float(np.degrees(np.arctan(m)))

    m_top, c_top, a_top = fit(all_xs[top_mask], all_tops[top_mask])
    m_bot, c_bot, a_bot = fit(all_xs[bot_mask], all_bots[bot_mask])

    # Scale back
    c_top /= scale
    c_bot /= scale
    orig_w = int(w / scale)

    top_y      = float(m_top * (orig_w/2) + c_top)
    bot_y      = float(m_bot * (orig_w/2) + c_bot)
    m_avg      = (m_top + m_bot) / 2
    face_width = abs(c_bot - c_top) / np.sqrt(m_avg**2 + 1)

    return {
        'top_y'         : top_y,
        'bot_y'         : bot_y,
        'm_top'         : m_top,
        'c_top'         : c_top,
        'm_bot'         : m_bot,
        'c_bot'         : c_bot,
        'angle_top_deg' : a_top,
        'angle_bot_deg' : a_bot,
        'face_angle_deg': (a_top + a_bot) / 2,
        'face_width_px' : float(face_width),
        'x_start'       : int(all_xs[top_mask].min() / scale),
        'x_end'         : int(all_xs[top_mask].max() / scale),
        'scale'         : scale,
    }


def find_shaft(gray, face_edges, px_per_mm=1.0):
    """
    Finds shaft above the top gear face edge.

    The shaft appears as a near-vertical dark or bright column
    just above the top gear face edge. We use CLAHE-enhanced
    Hough lines restricted to the shaft region.

    Parameters:
        gray        - grayscale image
        face_edges  - output from find_face_edges
        px_per_mm   - calibration factor

    Returns dict with shaft measurements or None if not detected.
    """
    h, w    = gray.shape
    top_y   = int(face_edges['top_y'])
    x_start = face_edges['x_start']
    x_end   = face_edges['x_end']

    # Shaft region: above top gear face, within gear x extent
    shaft_bottom = max(0, int(top_y*0.95))
    if shaft_bottom < 20:
        return None

    shaft_region = gray[:shaft_bottom, :]

    # CLAHE to enhance dark shaft
    clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(shaft_region)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), sigmaX=0)
    edges    = cv2.Canny(blurred, threshold1=10, threshold2=40)

    # Restrict to centre 60% of gear x span
    # Shaft is always near horizontal centre of gear
    gear_centre_x = (x_start + x_end) // 2
    gear_half_w   = (x_end - x_start) // 2
    shaft_x_start = max(0, gear_centre_x - int(gear_half_w * 0.6))
    shaft_x_end   = min(w, gear_centre_x + int(gear_half_w * 0.6))

    mask = np.zeros_like(edges)
    mask[:, shaft_x_start:shaft_x_end] = 255
    edges = cv2.bitwise_and(edges, mask)

    # Scale min line length to image size
    shaft_h = shaft_bottom
    min_len = max(15, int(shaft_h * 0.15))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_len,
        maxLineGap=20
    )

    if lines is None:
        return None

    # Filter near-vertical lines (70°–110°) close to top edge
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180
        if not (70 <= angle <= 110):
            continue
        y_mid = (y1 + y2) / 2
        if y_mid > shaft_bottom:
            continue
        candidates.append({
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'angle': angle,
            'x_mid': (x1 + x2) / 2,
        })

    if len(candidates) < 2:
        return None

    # Cluster by x — shaft lines share similar x
    x_mids = np.array([c['x_mid'] for c in candidates])
    x_med  = np.median(x_mids)
    x_std  = np.std(x_mids)

    # For shaft: lines should cluster tightly in x
    # Use tight tolerance — shaft width is small
    tight_tol = max(x_std * 0.8, 20)

    cluster = [c for c in candidates
               if abs(c['x_mid'] - x_med) < tight_tol]

    # If cluster is too small, relax tolerance
    if len(cluster) < 2:
        cluster = candidates

    shaft_x_left   = float(min(c['x_mid'] for c in cluster))
    shaft_x_right  = float(max(c['x_mid'] for c in cluster))
    shaft_x_centre = float(np.median([c['x_mid'] for c in cluster]))
    shaft_angle    = float(np.median([c['angle'] for c in cluster]))
    shaft_width_px = shaft_x_right - shaft_x_left

    # Perpendicularity
    face_angle     = face_edges['face_angle_deg']
    angle_between  = abs(shaft_angle - face_angle)
    if angle_between > 90:
        angle_between = 180 - angle_between
    perp_deviation = abs(angle_between - 90.0)

    return {
        'shaft_x_left'      : shaft_x_left,
        'shaft_x_right'     : shaft_x_right,
        'shaft_x_centre'    : shaft_x_centre,
        'shaft_width_px'    : shaft_width_px,
        'shaft_width_mm'    : shaft_width_px / px_per_mm,
        'shaft_angle_deg'   : shaft_angle,
        'perp_deviation_deg': perp_deviation,
        'candidates'        : candidates,
        'cluster'           : cluster,
    }


def find_helix_angle(gray, face_edges, gear_type='helical'):
    """
    Measures the helix angle from diagonal tooth lines on the gear face.

    For spur gears: returns 0° (no helix).
    For helical gears: detects diagonal Hough lines on the gear face
    and computes the helix angle from the vertical axis.

    Parameters:
        gray        - grayscale image
        face_edges  - output from find_face_edges
        gear_type   - 'spur' or 'helical'

    Returns dict with helix angle measurements.
    """
    if gear_type == 'spur':
        return {
            'helix_angle_deg'     : 0.0,
            'helix_angle_std_deg' : 0.0,
            'line_count'          : 0,
            'gear_type'           : 'spur',
        }

    h, w    = gray.shape
    top_y   = int(face_edges['top_y'])
    bot_y   = int(face_edges['bot_y'])
    x_start = face_edges['x_start']
    x_end   = face_edges['x_end']

    # ── Extract gear face ROI ──────────────────────────────────────────────
    margin  = 5
    roi_top = max(0, top_y + margin)    # slightly inside top edge
    roi_bot = min(h, bot_y - margin)    # slightly inside bottom edge

    if roi_bot <= roi_top:
        return {'helix_angle_deg': None, 'error': 'ROI too narrow'}

    # Additional safety — only use top 80% of face height
    # Bottom of face near base often has noise
    face_h    = roi_bot - roi_top
    roi_bot   = min(roi_bot, roi_top + int(face_h * 0.85))
    roi       = gray[roi_top:roi_bot, x_start:x_end]
    
    # ── Canny on ROI ───────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(roi, (3, 3), sigmaX=0)
    edges   = cv2.Canny(blurred, threshold1=30, threshold2=90)

    # ── Hough line detection ───────────────────────────────────────────────
    roi_h   = roi_bot - roi_top
    min_len = max(15, int(roi_h * 0.3))   # at least 30% of face height

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min_len,
        maxLineGap=8
    )

    if lines is None:
        return {'helix_angle_deg': None, 'error': 'No lines detected'}

    # ── Filter diagonal lines ──────────────────────────────────────────────
    # Helix lines are diagonal — exclude near-horizontal and near-vertical
    # Angle is measured from horizontal (0° = horizontal)
    # Helix lines will be between 10° and 80° from horizontal
    diag_angles = []
    diag_lines  = []
    noise_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        if 10 < abs(angle) < 80:
            diag_angles.append(float(angle))
            diag_lines.append(line[0])
        else:
            noise_lines.append(line[0])

    if len(diag_angles) == 0:
        return {'helix_angle_deg': None,
                'error': 'No diagonal lines found'}

    # ── Cluster dominant angle ─────────────────────────────────────────────
    # Remove outliers — keep angles within 2 std of median
    angles_arr = np.array(diag_angles)
    med        = np.median(angles_arr)
    std        = np.std(angles_arr)
    inliers    = angles_arr[np.abs(angles_arr - med) < 2 * std]

    mean_angle = float(np.mean(inliers))
    std_angle  = float(np.std(inliers))

    # ── Convert to helix angle ─────────────────────────────────────────────
    # Hough angle is measured from horizontal.
    # Helix angle is measured from the gear AXIS (vertical in side view).
    # helix_angle = 90° - abs(mean_angle_from_horizontal)
    helix_angle = 90.0 - abs(mean_angle)

    return {
        'helix_angle_deg'     : float(helix_angle),
        'helix_angle_std_deg' : std_angle,
        'hough_angle_deg'     : float(mean_angle),
        'line_count'          : len(inliers),
        'total_lines'         : len(diag_angles),
        'gear_type'           : 'helical',
        'diag_lines'          : diag_lines,
        'noise_lines'         : noise_lines,
        'roi_top'             : roi_top,
        'roi_bot'             : roi_bot,
        'x_start'             : x_start,
        'x_end'               : x_end,
    }


def run_sideprofile_pipeline(image_path, gear_type='helical',
                              px_per_mm=1.0):
    """
    Full side profile measurement pipeline.

    Parameters:
        image_path - path to side profile image
        gear_type  - 'spur' or 'helical'
        px_per_mm  - calibration factor

    Returns:
        measurements - dict of all measurements
        debug        - dict of intermediate data for visualisation
    """
    img  = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Face edges ─────────────────────────────────────────────────
    face  = find_face_edges(gray)

    # ── Step 2: Shaft ──────────────────────────────────────────────────────
    shaft = find_shaft(gray, face, px_per_mm)

    # ── Step 3: Helix angle ────────────────────────────────────────────────
    helix = find_helix_angle(gray, face, gear_type)

    # ── Assemble measurements ──────────────────────────────────────────────
    def px2mm(px): return px / px_per_mm

    measurements = {
        # Face width
        'face_width_px'       : face['face_width_px'],
        'face_width_mm'       : px2mm(face['face_width_px']),
        'face_angle_deg'      : face['face_angle_deg'],

        # Shaft
        'shaft_width_px'      : shaft['shaft_width_px']
                                if shaft else None,
        'shaft_width_mm'      : shaft['shaft_width_mm']
                                if shaft else None,
        'shaft_angle_deg'     : shaft['shaft_angle_deg']
                                if shaft else None,
        'perp_deviation_deg'  : shaft['perp_deviation_deg']
                                if shaft else None,

        # Helix
        'helix_angle_deg'     : helix.get('helix_angle_deg'),
        'helix_angle_std_deg' : helix.get('helix_angle_std_deg'),
        'helix_line_count'    : helix.get('line_count', 0),
        'gear_type'           : gear_type,
    }

    debug = {
        'gray'   : gray,
        'img'    : img,
        'face'   : face,
        'shaft'  : shaft,
        'helix'  : helix,
    }

    return measurements, debug


def print_sideprofile_summary(measurements):
    """Prints formatted side profile measurement summary."""
    print(f"── Side profile measurements ────────────────────────")
    print(f"  Face width        : "
          f"{measurements['face_width_px']:6.1f} px  "
          f"= {measurements['face_width_mm']:6.2f} mm")
    print(f"  Face tilt         : "
          f"{measurements['face_angle_deg']:6.2f}°")

    if measurements['shaft_width_px'] is not None:
        print(f"  Shaft width       : "
              f"{measurements['shaft_width_px']:6.1f} px  "
              f"= {measurements['shaft_width_mm']:6.2f} mm")
        print(f"  Shaft angle       : "
              f"{measurements['shaft_angle_deg']:6.2f}°")
        print(f"  Perp deviation    : "
              f"{measurements['perp_deviation_deg']:6.2f}°  "
              f"({'OK' if measurements['perp_deviation_deg'] < 2 else 'CHECK'})")
    else:
        print(f"  Shaft             : not detected")

    if measurements['helix_angle_deg'] is not None:
        print(f"  Helix angle       : "
              f"{measurements['helix_angle_deg']:6.2f}°  "
              f"± {measurements['helix_angle_std_deg']:.2f}°  "
              f"({measurements['helix_line_count']} lines)")
    else:
        print(f"  Helix angle       : not detected")
    print()