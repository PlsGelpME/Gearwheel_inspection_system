import cv2
import numpy as np
import os
import json
from pipeline.sideProfile import run_sideprofile_pipeline

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
IMG_DIR   = "images/tooth_face"
RESULTS   = "results"
PX_PER_MM = 20.0
GEAR_TYPE = "helical"   # "spur" or "helical"

# Only process files matching this prefix
# Change to "toothface" if your side profile images use that name
# Or set to "" to process all images in the folder
IMG_PREFIX = ""         # ← set to filename prefix e.g. "side_" or "toothface"

os.makedirs(RESULTS, exist_ok=True)

# ══════════════════════════════════════════════════════
# COLLECT IMAGES
# ══════════════════════════════════════════════════════
all_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    and f.startswith(IMG_PREFIX)
])

if len(all_files) == 0:
    print(f"No images found in {IMG_DIR}/ with prefix '{IMG_PREFIX}'")
    exit()

print(f"Found {len(all_files)} images")
print(f"Gear type  : {GEAR_TYPE}")
print(f"PX_PER_MM  : {PX_PER_MM}")
print()

# ══════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════
all_results = []

# Print header
print(f"{'Image':<25} "
      f"{'FW px':>7} "
      f"{'FW mm':>7} "
      f"{'Tilt°':>7} "
      f"{'Shaft mm':>9} "
      f"{'Perp°':>7} "
      f"{'Helix°':>8} "
      f"{'Hx std':>7} "
      f"{'Lines':>6} "
      f"{'Status':>8}")
print("─" * 102)

for fname in all_files:
    path   = os.path.join(IMG_DIR, fname)
    result = {
        'filename' : fname,
        'status'   : 'OK',
        'error'    : None
    }

    try:
        m, debug = run_sideprofile_pipeline(path, GEAR_TYPE, PX_PER_MM)

        # ── Validation flags ───────────────────────────────────────────────
        flags = []

        if m['face_width_px'] < 20:
            flags.append("fw_too_small")

        if m['perp_deviation_deg'] is not None and \
           m['perp_deviation_deg'] > 5:
            flags.append(f"perp={m['perp_deviation_deg']:.1f}°")

        if m['helix_angle_deg'] is None:
            flags.append("helix_not_detected")
        elif m['helix_angle_std_deg'] > 5:
            flags.append("helix_noisy")

        status = "WARN" if flags else "OK"

        # ── Store ──────────────────────────────────────────────────────────
        result.update({
            'face_width_px'      : m['face_width_px'],
            'face_width_mm'      : m['face_width_mm'],
            'face_angle_deg'     : m['face_angle_deg'],
            'shaft_width_mm'     : m['shaft_width_mm'],
            'perp_deviation_deg' : m['perp_deviation_deg'],
            'helix_angle_deg'    : m['helix_angle_deg'],
            'helix_angle_std'    : m['helix_angle_std_deg'],
            'helix_line_count'   : m['helix_line_count'],
            'status'             : status,
            'flags'              : flags,
        })

        # ── Print row ──────────────────────────────────────────────────────
        fw_px   = m['face_width_px']
        fw_mm   = m['face_width_mm']
        tilt    = m['face_angle_deg']
        sh_mm   = m['shaft_width_mm']   if m['shaft_width_mm']   else 0
        perp    = m['perp_deviation_deg'] if m['perp_deviation_deg'] else 0
        hx      = m['helix_angle_deg']  if m['helix_angle_deg']  else 0
        hx_std  = m['helix_angle_std_deg'] if m['helix_angle_std_deg'] else 0
        hx_n    = m['helix_line_count']

        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        print(f"{fname:<25} "
              f"{fw_px:>7.1f} "
              f"{fw_mm:>7.2f} "
              f"{tilt:>7.2f} "
              f"{sh_mm:>9.2f} "
              f"{perp:>7.2f} "
              f"{hx:>8.2f} "
              f"{hx_std:>7.2f} "
              f"{hx_n:>6} "
              f"{status:>8}"
              + flag_str)

    except Exception as e:
        result['status'] = 'ERROR'
        result['error']  = str(e)
        print(f"{fname:<25} ERROR: {e}")

    all_results.append(result)

# ══════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════
good = [r for r in all_results
        if r['status'] in ('OK', 'WARN')
        and r.get('face_width_px') is not None]

errors = [r for r in all_results if r['status'] == 'ERROR']

print()
print("─" * 102)
print(f"Processed : {len(all_results)}  "
      f"OK/WARN : {len(good)}  "
      f"ERROR : {len(errors)}")

if len(good) > 0:
    def stats(key):
        vals = [r[key] for r in good
                if r.get(key) is not None]
        if not vals:
            return None, None, None, None
        return (float(np.mean(vals)),
                float(np.std(vals)),
                float(min(vals)),
                float(max(vals)))

    print()
    print(f"── Summary across {len(good)} images ──────────────────────────────")
    print(f"{'Measurement':<25} {'Mean':>9} {'Std':>8} "
          f"{'Min':>9} {'Max':>9}")
    print("─" * 65)

    metrics = [
        ('Face width (px)',   'face_width_px'),
        ('Face width (mm)',   'face_width_mm'),
        ('Face tilt (°)',     'face_angle_deg'),
        ('Shaft width (mm)',  'shaft_width_mm'),
        ('Perp deviation (°)','perp_deviation_deg'),
        ('Helix angle (°)',   'helix_angle_deg'),
        ('Helix std (°)',     'helix_angle_std'),
        ('Helix line count',  'helix_line_count'),
    ]

    for label, key in metrics:
        mean, std, lo, hi = stats(key)
        if mean is None:
            print(f"{label:<25} {'N/A':>9}")
            continue
        print(f"{label:<25} {mean:>9.3f} {std:>8.3f} "
              f"{lo:>9.3f} {hi:>9.3f}")

    # Helix angle histogram in console
    helix_vals = [r['helix_angle_deg'] for r in good
                  if r.get('helix_angle_deg') is not None]
    if helix_vals:
        print()
        print(f"── Helix angle distribution ──────────────────────────────")
        bins   = np.arange(0, 90, 5)
        counts, edges = np.histogram(helix_vals, bins=bins)
        for i, count in enumerate(counts):
            if count > 0:
                bar = "▓" * count
                print(f"  {edges[i]:4.0f}°–{edges[i+1]:4.0f}° : "
                      f"{bar} ({count})")

    # Flag summary
    all_flags = []
    for r in good:
        all_flags.extend(r.get('flags', []))
    if all_flags:
        from collections import Counter
        print()
        print(f"── Warnings summary ──────────────────────────────────────")
        for flag, count in Counter(all_flags).most_common():
            print(f"  {flag:<30} : {count} images")

# ══════════════════════════════════════════════════════
# SAVE JSON
# ══════════════════════════════════════════════════════
out = os.path.join(RESULTS, 'batch_sideprofile.json')
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {out}")