import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pipeline.gear_core      import run_core_pipeline
from pipeline.gear_mask      import build_gear_mask
from pipeline.tooth_analysis import (get_contour_signal,
                                      detect_teeth_from_contour,
                                      measure_teeth_from_contour,
                                      print_tooth_summary)

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
IMG_DIR        = "images/real_time_gear"
RESULTS_DIR    = "results"
PX_PER_MM      = 20.0          # adjust after calibration
EXPECTED_TEETH = None            # set to None for auto-detect
GEAR_TYPE      = "spur"        # "spur" or "helical"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# COLLECT IMAGES
# ══════════════════════════════════════════════════════
images = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
])

if len(images) == 0:
    print("No images found in images/ folder")
    exit()

print(f"Found {len(images)} images")
print(f"PX_PER_MM      : {PX_PER_MM}")
print(f"Expected teeth : {EXPECTED_TEETH}")
print(f"Gear type      : {GEAR_TYPE}")
print()

# ══════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════
all_results = []

# Print table header
print(f"{'Image':<20} "
      f"{'Cx':>5} {'Cy':>5} "
      f"{'Shaft':>7} "
      f"{'Tip D':>7} "
      f"{'Root D':>7} "
      f"{'Pitch D':>8} "
      f"{'Teeth':>6} "
      f"{'TW px':>7} "
      f"{'TW sd':>6} "
      f"{'TD px':>7} "
      f"{'Circ':>6} "
      f"{'Status':>8}")
print("─" * 115)

for fname in images:
    path   = os.path.join(IMG_DIR, fname)
    result = {'filename': fname, 'status': 'OK', 'error': None}

    try:
        # ── Core pipeline ──────────────────────────────────────────────────
        m, inter = run_core_pipeline(path, px_per_mm=PX_PER_MM)

        gray    = inter['gray']
        gcx     = m['gear_centre_x']
        gcy     = m['gear_centre_y']
        tip_r   = inter['tip_radius_px']
        root_r  = inter['root_radius_px']
        shaft_r = inter['shaft_radius_px']

        # ── Tooth analysis ─────────────────────────────────────────────────
        gear_only, _ = build_gear_mask(
            gray, gcx, gcy, shaft_r, tip_r)

        angles, dists, pts = get_contour_signal(gear_only, gcx, gcy)

        tooth_peaks, gap_valleys, smoothed = detect_teeth_from_contour(
            dists, angles, expected_count=EXPECTED_TEETH)

        measurements, summary = measure_teeth_from_contour(
            dists, angles, pts,
            tooth_peaks, gap_valleys, smoothed,
            gcx, gcy, PX_PER_MM)

        # ── Extract key values ─────────────────────────────────────────────
        tooth_count   = summary.get('tooth_count', 0)
        tip_r_fit     = summary.get('tip_radius_fitted_px',  tip_r)
        root_r_fit    = summary.get('root_radius_fitted_px', root_r)
        pitch_r_fit   = tip_r_fit - (tip_r_fit - root_r_fit) / 2.25
        shaft_d       = m['shaft_diameter_px']
        tip_d         = tip_r_fit  * 2
        root_d        = root_r_fit * 2
        pitch_d       = pitch_r_fit * 2
        tooth_w_mean  = summary.get('tooth_width_mean_px',  0)
        tooth_w_std   = summary.get('tooth_width_std_px',   0)
        tooth_d_mean  = summary.get('tooth_depth_mean_px',  0)
        circularity   = m['circularity']
        circularity_gdt_zone = m['circularity_gdt_zone']
        circularity_gdt_std = m['circularity_gdt_std']

        # ── Validation flags ───────────────────────────────────────────────
        flags = []
        if EXPECTED_TEETH and tooth_count != EXPECTED_TEETH:
            flags.append(f"teeth={tooth_count}≠{EXPECTED_TEETH}")
        if tooth_w_std > 5:
            flags.append("TW_std_high")
        if circularity < 0.5:
            flags.append("circ_low")

        status = "WARN" if flags else "OK"
        gdt_zone    = m.get('circularity_gdt_zone', 0)

                # ── Store result ───────────────────────────────────────────────────
        result.update({
            'gear_cx'           : gcx,
            'gear_cy'           : gcy,
            'shaft_diameter_px' : shaft_d,
            'shaft_diameter_mm' : shaft_d / PX_PER_MM,
            'tip_diameter_px'   : tip_d,
            'tip_diameter_mm'   : tip_d / PX_PER_MM,
            'root_diameter_px'  : root_d,
            'root_diameter_mm'  : root_d / PX_PER_MM,
            'pitch_diameter_px' : pitch_d,
            'pitch_diameter_mm' : pitch_d / PX_PER_MM,
            'tooth_count'       : tooth_count,
            'tooth_width_px'    : tooth_w_mean,
            'tooth_width_mm'    : tooth_w_mean / PX_PER_MM,
            'tooth_width_std_px': tooth_w_std,
            'tooth_depth_px'    : tooth_d_mean,
            'tooth_depth_mm'    : tooth_d_mean / PX_PER_MM,
            'circularity'       : circularity,
            'circularity_gdt_zone'  : circularity_gdt_zone,
            'circularity_gdt_std'   : circularity_gdt_std,
            'status'            : status,
            'flags'             : flags,
        })

        

        print(f"{fname:<20} "
              f"{gcx:>5} {gcy:>5} "
              f"{shaft_d:>7.1f} "
              f"{tip_d:>7.1f} "
              f"{root_d:>7.1f} "
              f"{pitch_d:>8.1f} "
              f"{tooth_count:>6} "
              f"{tooth_w_mean:>7.1f} "
              f"{tooth_w_std:>6.1f} "
              f"{tooth_d_mean:>7.1f} "
              f"{circularity:>6.3f} "
              f"{gdt_zone:>8.1f} "
              f"{status:>6}"
              + (f"  [{', '.join(flags)}]" if flags else ""))

    except Exception as e:
        result['status'] = 'ERROR'
        result['error']  = str(e)
        print(f"{fname:<20} ERROR: {e}")

    all_results.append(result)

# ══════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════
good = [r for r in all_results if r['status'] in ('OK', 'WARN')
        and 'tip_diameter_px' in r]

print()
print("─" * 115)
print(f"Processed : {len(all_results)}  "
      f"OK/WARN : {len(good)}  "
      f"ERROR : {len(all_results)-len(good)}")

if len(good) > 0:
    def stats(key):
        vals = [r[key] for r in good]
        return np.mean(vals), np.std(vals), min(vals), max(vals)

    print()
    print(f"── Summary across {len(good)} good images ──────────────────────────")
    print(f"{'Measurement':<22} {'Mean':>10} {'Std':>8} "
          f"{'Min':>10} {'Max':>10}")
    print("─" * 65)

    metrics = [
        ('Circ ISO',         'circularity'),
        ('Circ GD&T zone px','circularity_gdt_zone'),
        ('Shaft diam (px)',  'shaft_diameter_px'),
        ('Tip diam (px)',    'tip_diameter_px'),
        ('Root diam (px)',   'root_diameter_px'),
        ('Pitch diam (px)',  'pitch_diameter_px'),
        ('Tooth count',      'tooth_count'),
        ('Tooth width (px)', 'tooth_width_px'),
        ('Tooth width std',  'tooth_width_std_px'),
        ('Tooth depth (px)', 'tooth_depth_px'),
    ]

    for label, key in metrics:
        mean, std, lo, hi = stats(key)
        print(f"{label:<22} {mean:>10.2f} {std:>8.2f} "
              f"{lo:>10.2f} {hi:>10.2f}")

    print()
    print(f"── Measurements in mm (px_per_mm={PX_PER_MM}) ──────────────────")
    print(f"{'Measurement':<22} {'Mean mm':>10} {'Std mm':>8}")
    print("─" * 45)

    mm_metrics = [
        ('Shaft diameter',  'shaft_diameter_mm'),
        ('Tip diameter',    'tip_diameter_mm'),
        ('Root diameter',   'root_diameter_mm'),
        ('Pitch diameter',  'pitch_diameter_mm'),
        ('Tooth width',     'tooth_width_mm'),
        ('Tooth depth',     'tooth_depth_mm'),
    ]

    for label, key in mm_metrics:
        mean, std, _, _ = stats(key)
        print(f"{label:<22} {mean:>10.3f} {std:>8.3f}")

# ══════════════════════════════════════════════════════
# SAVE RESULTS TO JSON
# ══════════════════════════════════════════════════════
name = 'RTGbatch_results.json'
out_path = os.path.join(RESULTS_DIR, name)
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nFull results saved to {out_path}")