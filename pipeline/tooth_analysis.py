import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def get_contour_signal(gear_only_mask, gear_cx, gear_cy,
                       tip_radius_px=None):
    """
    Extracts the gear outer contour and converts to polar signal.
    Optionally filters points outside the tip radius to remove
    fragments of other gears visible in the frame.

    Returns:
        angles  - sorted angles in degrees (0-360)
        dists   - radial distances at each angle
        pts     - original (x,y) contour points sorted by angle
    """
    contours, _ = cv2.findContours(
        gear_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise RuntimeError("No contours found in gear mask")

    # ── Find the contour closest to gear centre ────────────────────────────
    # This handles cases where other gear fragments appear as separate
    # white regions in the mask
    best_contour = None
    best_dist    = float('inf')

    for cnt in contours:
        M    = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        ccx  = M['m10'] / M['m00']
        ccy  = M['m01'] / M['m00']
        dist = np.sqrt((ccx - gear_cx)**2 + (ccy - gear_cy)**2)
        if dist < best_dist:
            best_dist    = dist
            best_contour = cnt

    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    pts  = best_contour.reshape(-1, 2).astype(float)

    # Convert to polar
    dx   = pts[:, 0] - gear_cx
    dy   = pts[:, 1] - gear_cy
    dist = np.sqrt(dx**2 + dy**2)
    ang  = np.degrees(np.arctan2(dy, dx)) % 360

    # ── Filter out points beyond tip radius ───────────────────────────────
    # Removes stray fragments from other gears in the frame
    if tip_radius_px is not None:
        # Allow 5% beyond tip for tooth tip variation
        max_dist = tip_radius_px * 1.05
        valid    = dist <= max_dist
        dist     = dist[valid]
        ang      = ang[valid]
        pts      = pts[valid]

    # Sort by angle
    idx  = np.argsort(ang)
    return ang[idx], dist[idx], pts[idx]


def detect_teeth_from_contour(dists, angles,
                               expected_count=None,
                               smoothing_window=7):
    """
    Detects tooth peaks and gap valleys from contour distance signal.
    Uses signal rotation to avoid missing features at 0/360 boundary.
    """
    n   = len(dists)
    win = smoothing_window if smoothing_window % 2 == 1 \
          else smoothing_window + 1
    win = min(win, n - 1)

    # ── Estimate pitch in samples ──────────────────────────────────────────
    if expected_count is not None:
        pitch_samples = n / expected_count
    else:
        pitch_samples = n / 100   # conservative default

    min_dist   = max(int(pitch_samples * 0.5), 3)
    sig_range  = dists.max() - dists.min()
    prominence = sig_range * 0.15

    # ── Rotate signal by quarter pitch before peak detection ───────────────
    # Moves the 0/360 seam to the middle of a tooth gap — safe territory
    quarter_shift = int(pitch_samples * 0.25)
    rotated       = np.roll(dists, -quarter_shift)
    smoothed_rot  = savgol_filter(rotated, window_length=win, polyorder=3)

    # ── Detect peaks on rotated signal ────────────────────────────────────
    raw_peaks, _ = find_peaks(
        smoothed_rot,
        distance=min_dist,
        prominence=prominence
    )

    # ── Rotate by three-quarter pitch for valley detection ─────────────────
    # Different offset ensures valleys aren't at seam either
    three_q_shift = int(pitch_samples * 0.75)
    rotated_v     = np.roll(dists, -three_q_shift)
    smoothed_v    = savgol_filter(rotated_v, window_length=win, polyorder=3)

    raw_valleys, _ = find_peaks(
        -smoothed_v,
        distance=min_dist,
        prominence=prominence
    )

    # ── Rotate indices back to original positions ──────────────────────────
    tooth_peaks = np.sort((raw_peaks   + quarter_shift) % n)
    gap_valleys = np.sort((raw_valleys + three_q_shift) % n)

    # ── Smooth the original signal for measurement ─────────────────────────
    smoothed = savgol_filter(dists, window_length=win, polyorder=3)

    return tooth_peaks, gap_valleys, smoothed


def measure_teeth_from_contour(dists, angles, pts,
                                tooth_peaks, gap_valleys,
                                smoothed, gear_cx, gear_cy,
                                px_per_mm=1.0):
    """
    Measures tooth geometry from contour signal.

    Fits best circles to tip points and valley points.
    Snaps any tip point below the fitted tip circle outward to it.
    """
    from pipeline.gear_core import fit_circle_to_points

    n         = len(dists)
    num_teeth = len(tooth_peaks)

    # ── Collect raw tip and valley coordinates ─────────────────────────────
    tip_pts     = []
    valley_pts  = []

    for peak_idx in tooth_peaks:
        tip_pts.append((float(pts[peak_idx][0]),
                        float(pts[peak_idx][1])))

    for val_idx in gap_valleys:
        valley_pts.append((float(pts[val_idx][0]),
                           float(pts[val_idx][1])))

    # ── Fit circles ────────────────────────────────────────────────────────
    tip_cx,  tip_cy,  tip_r_fit  = fit_circle_to_points(tip_pts)
    root_cx, root_cy, root_r_fit = fit_circle_to_points(valley_pts)

    '''
    print(f"  Fitted tip circle  : centre=({tip_cx:.1f},{tip_cy:.1f})  "
          f"r={tip_r_fit:.1f}px")
    print(f"  Fitted root circle : centre=({root_cx:.1f},{root_cy:.1f})  "
          f"r={root_r_fit:.1f}px")
    '''
    # ── Snap tip points to fitted tip circle ───────────────────────────────
    snapped_tips = []
    snap_count   = 0

    for tx, ty in tip_pts:
        dx   = tx - tip_cx
        dy   = ty - tip_cy
        dist = np.sqrt(dx**2 + dy**2)

        if dist < tip_r_fit:
            # Point is inside fitted tip circle — snap outward
            scale = tip_r_fit / dist
            tx    = tip_cx + dx * scale
            ty    = tip_cy + dy * scale
            snap_count += 1

        snapped_tips.append((tx, ty))

    if snap_count > 0:
        print(f"  Snapped {snap_count} tip points to tip circle")

    # ── Measure each tooth ─────────────────────────────────────────────────
    measurements = []

    for i, peak_idx in enumerate(tooth_peaks):

        # Use snapped tip position
        tip_x, tip_y = snapped_tips[i]
        tip_angle    = float(angles[peak_idx])

        # Find nearest valley on each side
        left_valleys  = gap_valleys[gap_valleys < peak_idx]
        right_valleys = gap_valleys[gap_valleys > peak_idx]

        if len(left_valleys) == 0:
            left_val_idx = gap_valleys[-1]
        else:
            left_val_idx = left_valleys[-1]

        if len(right_valleys) == 0:
            right_val_idx = gap_valleys[0]
        else:
            right_val_idx = right_valleys[0]

        left_x,  left_y  = float(pts[left_val_idx][0]), \
                            float(pts[left_val_idx][1])
        right_x, right_y = float(pts[right_val_idx][0]), \
                            float(pts[right_val_idx][1])

        # ── Tooth width — chord between left and right valleys ─────────────
        tooth_width_px = float(np.sqrt(
            (right_x - left_x)**2 + (right_y - left_y)**2))

        # ── Tooth depth — snapped tip to root circle surface ───────────────
        # Distance from tip to gear centre minus root radius
        tip_from_centre = np.sqrt(
            (tip_x - gear_cx)**2 + (tip_y - gear_cy)**2)
        tooth_depth_px  = float(tip_from_centre - root_r_fit)

        # ── Gap width ──────────────────────────────────────────────────────
        next_peak_idx = tooth_peaks[(i + 1) % num_teeth]
        next_left_vals = gap_valleys[gap_valleys < next_peak_idx]
        if len(next_left_vals) == 0:
            next_left_val_idx = gap_valleys[-1]
        else:
            next_left_val_idx = next_left_vals[-1]

        next_left_x = float(pts[next_left_val_idx][0])
        next_left_y = float(pts[next_left_val_idx][1])

        gap_width_px = float(np.sqrt(
            (next_left_x - right_x)**2 +
            (next_left_y - right_y)**2))

        # ── Flank angles ───────────────────────────────────────────────────
        radial_angle = np.arctan2(tip_y - gear_cy, tip_x - gear_cx)

        lead_vec    = np.arctan2(tip_y  - left_y,  tip_x  - left_x)
        angle_lead  = float(abs(np.degrees(
            lead_vec - radial_angle)) % 90)

        trail_vec   = np.arctan2(right_y - tip_y,  right_x - tip_x)
        angle_trail = float(abs(np.degrees(
            trail_vec - radial_angle)) % 90)

        symmetry = abs(angle_lead - angle_trail)

        measurements.append({
            'tooth_idx'      : i,
            'tip_angle_deg'  : tip_angle,
            'tip_x'          : tip_x,
            'tip_y'          : tip_y,
            'left_x'         : left_x,
            'left_y'         : left_y,
            'right_x'        : right_x,
            'right_y'        : right_y,
            'tooth_width_px' : tooth_width_px,
            'tooth_depth_px' : tooth_depth_px,
            'gap_width_px'   : gap_width_px,
            'angle_lead_deg' : angle_lead,
            'angle_trail_deg': angle_trail,
            'symmetry_deg'   : symmetry,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    if len(measurements) == 0:
        return measurements, {}

    def px2mm(px): return px / px_per_mm

    widths  = [m['tooth_width_px']  for m in measurements]
    depths  = [m['tooth_depth_px']  for m in measurements]
    gaps    = [m['gap_width_px']    for m in measurements]
    leads   = [m['angle_lead_deg']  for m in measurements]
    trails  = [m['angle_trail_deg'] for m in measurements]
    syms    = [m['symmetry_deg']    for m in measurements]

    summary = {
        'tooth_count'          : len(measurements),

        'tip_circle_cx'        : tip_cx,
        'tip_circle_cy'        : tip_cy,
        'tip_radius_fitted_px' : tip_r_fit,
        'tip_diameter_fitted_px': tip_r_fit * 2,

        'root_circle_cx'       : root_cx,
        'root_circle_cy'       : root_cy,
        'root_radius_fitted_px': root_r_fit,
        'root_diameter_fitted_px': root_r_fit * 2,

        'tooth_width_mean_px'  : float(np.mean(widths)),
        'tooth_width_std_px'   : float(np.std(widths)),
        'tooth_width_mean_mm'  : px2mm(np.mean(widths)),

        'tooth_depth_mean_px'  : float(np.mean(depths)),
        'tooth_depth_std_px'   : float(np.std(depths)),
        'tooth_depth_mean_mm'  : px2mm(np.mean(depths)),

        'gap_width_mean_px'    : float(np.mean(gaps)),
        'gap_width_mean_mm'    : px2mm(np.mean(gaps)),

        'angle_lead_mean_deg'  : float(np.mean(leads)),
        'angle_trail_mean_deg' : float(np.mean(trails)),
        'symmetry_mean_deg'    : float(np.mean(syms)),
        'symmetry_std_deg'     : float(np.std(syms)),
    }

    return measurements, summary


def run_tooth_analysis(gray, gear_cx, gear_cy,
                       tip_radius_px, root_radius_px,
                       pitch_radius_px, px_per_mm=1.0,
                       expected_count=None):
    """
    Full tooth analysis using gear contour signal.
    """
    from pipeline.gear_mask import build_gear_mask

    shaft_r = root_radius_px * 0.15

    # ── Build mask ─────────────────────────────────────────────────────────
    gear_only, gear_filled = build_gear_mask(
        gray, gear_cx, gear_cy, shaft_r, tip_radius_px)

    # ── Extract contour signal ─────────────────────────────────────────────
    angles, dists, pts = get_contour_signal(
        gear_only, gear_cx, gear_cy,
        tip_radius_px)   # ← add this
    
    # ── Detect teeth ───────────────────────────────────────────────────────
    tooth_peaks, gap_valleys, smoothed = detect_teeth_from_contour(
        dists, angles, expected_count=expected_count)

    '''
    print(f"  Contour points  : {len(pts)}")
    print(f"  Peaks detected  : {len(tooth_peaks)}")
    print(f"  Valleys detected: {len(gap_valleys)}")
    '''

    # ── Measure ────────────────────────────────────────────────────────────
    measurements, summary = measure_teeth_from_contour(
        dists, angles, pts,
        tooth_peaks, gap_valleys, smoothed,
        gear_cx, gear_cy, px_per_mm)

    results = {
        'tooth_count'          : summary.get('tooth_count', 0),
        'detect_radius_px'     : float(tip_radius_px),
        'pitch_radius_px'      : float(pitch_radius_px),
        'pitch_diameter_px'    : float(pitch_radius_px * 2),
        'pitch_diameter_mm'    : pitch_radius_px * 2 / px_per_mm,
        'tooth_width_mean_px'  : summary.get('tooth_width_mean_px'),
        'tooth_width_std_px'   : summary.get('tooth_width_std_px'),
        'tooth_width_mean_mm'  : summary.get('tooth_width_mean_mm'),
        'tooth_depth_mean_px'  : summary.get('tooth_depth_mean_px'),
        'tooth_depth_mean_mm'  : summary.get('tooth_depth_mean_mm'),
        'gap_width_mean_px'    : summary.get('gap_width_mean_px'),
        'gap_width_mean_mm'    : summary.get('gap_width_mean_mm'),
        'angle_lead_mean_deg'  : summary.get('angle_lead_mean_deg'),
        'angle_trail_mean_deg' : summary.get('angle_trail_mean_deg'),
        'symmetry_mean_deg'    : summary.get('symmetry_mean_deg'),
        'tip_radius_fitted_px'  : summary.get('tip_radius_fitted_px'),
        'tip_diameter_fitted_px': summary.get('tip_diameter_fitted_px'),
        'root_radius_fitted_px' : summary.get('root_radius_fitted_px'),
        'root_diameter_fitted_px': summary.get('root_diameter_fitted_px'),
        'root_circle_cx'        : summary.get('root_circle_cx'),
        'root_circle_cy'        : summary.get('root_circle_cy'),
        'per_tooth'            : measurements,
    }

    debug_data = {
        'angles'      : angles,
        'signal'      : dists,
        'smoothed'    : smoothed,
        'threshold'   : (dists.max() + dists.min()) / 2,
        'tooth_peaks' : tooth_peaks,
        'gap_valleys' : gap_valleys,
        'detect_r'    : tip_radius_px,
        'true_pitch_r': pitch_radius_px,
        'gear_only'   : gear_only,
        'pts'         : pts,
    }

    return results, debug_data


def print_tooth_summary(results, px_per_mm=1.0):
    """Prints formatted tooth measurement summary."""
    print(f"── Tooth analysis ───────────────────────────────────")
    print(f"  Tooth count       : {results['tooth_count']}")
    if results.get('tooth_width_mean_px'):
        print(f"  Tooth width       : "
              f"{results['tooth_width_mean_px']:6.1f} px  "
              f"± {results['tooth_width_std_px']:4.1f} px  "
              f"= {results['tooth_width_mean_mm']:5.2f} mm")
    if results.get('tooth_depth_mean_px'):
        print(f"  Tooth depth       : "
              f"{results['tooth_depth_mean_px']:6.1f} px  "
              f"= {results['tooth_depth_mean_mm']:5.2f} mm")
    if results.get('gap_width_mean_px'):
        print(f"  Gap width         : "
              f"{results['gap_width_mean_px']:6.1f} px  "
              f"= {results['gap_width_mean_mm']:5.2f} mm")
    if results.get('angle_lead_mean_deg'):
        print(f"  Lead flank angle  : "
              f"{results['angle_lead_mean_deg']:6.2f}°")
        print(f"  Trail flank angle : "
              f"{results['angle_trail_mean_deg']:6.2f}°")
        print(f"  Symmetry          : "
              f"{results['symmetry_mean_deg']:6.2f}°  "
              f"({'OK' if results['symmetry_mean_deg'] < 5 else 'CHECK'})")
    print()