import cv2
import numpy as np


def build_gear_mask(gray, gear_cx, gear_cy,
                    shaft_radius_px, tip_radius_px):
    """
    Builds a clean binary mask of the gear body.
    Uses ROI-based Otsu thresholding restricted to gear region.
    """
    h, w = gray.shape

    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)

    # ROI-based Otsu — only within tip circle
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (gear_cx, gear_cy),
               int(tip_radius_px) + 10, 255, -1)

    roi_pixels = blurred[roi_mask > 0]

    hist, _   = np.histogram(roi_pixels, bins=256, range=(0, 256))
    hist_norm = hist.astype(float) / hist.sum()

    best_thresh = 0
    best_var    = 0.0

    for t in range(1, 256):
        w0 = hist_norm[:t].sum()
        w1 = hist_norm[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (hist_norm[:t] * np.arange(t)).sum() / w0
        mu1 = (hist_norm[t:] * np.arange(t, 256)).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var    = var
            best_thresh = t

    _, binary = cv2.threshold(blurred, best_thresh, 255,
                               cv2.THRESH_BINARY)

    # Auto-orient — gear should be white
    mid_r  = int((shaft_radius_px + tip_radius_px) / 2)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    xs     = np.clip(
        (gear_cx + mid_r * np.cos(angles)).astype(int), 0, w-1)
    ys     = np.clip(
        (gear_cy + mid_r * np.sin(angles)).astype(int), 0, h-1)
    ring_mean = binary[ys, xs].mean()
    if ring_mean < 127:
        binary = cv2.bitwise_not(binary)

    # Find gear body component
    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(binary, connectivity=8)

    sample_r = int(shaft_radius_px +
                   (tip_radius_px - shaft_radius_px) * 0.5)
    sample_x = np.clip(int(gear_cx + sample_r), 0, w-1)
    sample_y = np.clip(int(gear_cy),             0, h-1)

    gear_label = int(labels[sample_y, sample_x])

    if gear_label == 0:
        for dx, dy in [(10,0),(-10,0),(0,10),(0,-10)]:
            lx = np.clip(sample_x+dx, 0, w-1)
            ly = np.clip(sample_y+dy, 0, h-1)
            if labels[ly, lx] != 0:
                gear_label = int(labels[ly, lx])
                break

    gear_component = np.zeros((h, w), dtype=np.uint8)
    if gear_label > 0:
        gear_component[labels == gear_label] = 255
    else:
        areas      = stats[1:, cv2.CC_STAT_AREA]
        gear_label = int(np.argmax(areas)) + 1
        gear_component[labels == gear_label] = 255

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gear_comp = cv2.morphologyEx(
        gear_component, cv2.MORPH_CLOSE, kernel, iterations=2)

    tip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(tip_mask, (gear_cx, gear_cy),
               int(tip_radius_px) + 5, 255, -1)
    gear_comp = cv2.bitwise_and(gear_comp, tip_mask)

    gear_only   = gear_comp.copy()
    gear_filled = gear_comp.copy()
    cv2.circle(gear_filled, (gear_cx, gear_cy),
               int(shaft_radius_px) - 2, 255, -1)

    return gear_only, gear_filled


def radial_scan_binary(mask, gear_cx, gear_cy,
                       start_radius, end_radius,
                       num_angles=1440):
    """
    Scans a binary mask radially from gear centre.

    At each radius samples the mask at num_angles positions.
    Returns the fraction of white pixels at each radius.

    Returns:
        radii       - array of radii scanned
        duty_cycles - fraction of white pixels at each radius (0-1)
    """
    h, w       = mask.shape
    angles     = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    cos_a      = np.cos(angles)
    sin_a      = np.sin(angles)

    radii      = np.arange(int(start_radius), int(end_radius) + 1)
    duties     = []

    for r in radii:
        xs   = np.clip(
            (gear_cx + r * cos_a).astype(int), 0, w-1)
        ys   = np.clip(
            (gear_cy + r * sin_a).astype(int), 0, h-1)
        vals = mask[ys, xs].astype(float)
        duties.append(vals.mean() / 255.0)

    return radii, np.array(duties)


def sample_ring_binary(mask, gear_cx, gear_cy,
                       radius, num_angles=1440):
    """
    Samples binary mask at a single radius.

    Returns:
        angles - degrees 0-360
        signal - 0.0 or 1.0 at each angle (gap or tooth)
    """
    h, w    = mask.shape
    angles  = np.linspace(0, 360, num_angles, endpoint=False)
    radians = np.radians(angles)

    xs = np.clip(
        (gear_cx + radius * np.cos(radians)).astype(int), 0, w-1)
    ys = np.clip(
        (gear_cy + radius * np.sin(radians)).astype(int), 0, h-1)

    signal = mask[ys, xs].astype(float) / 255.0
    return angles, signal