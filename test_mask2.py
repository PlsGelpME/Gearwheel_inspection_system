import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pipeline.gear_core import run_core_pipeline
from pipeline.gear_mask import (build_gear_mask,
                                 radial_scan_binary,
                                 sample_ring_binary)

IMG_DIR      = "images/end_face/Gear_2"
PX_PER_MM = 20.0
RESULTS_DIR    = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# COLLECT IMAGES
# ══════════════════════════════════════════════════════
images = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
])

for fname in images:
    path   = os.path.join(IMG_DIR, fname)
    result = {'filename': fname, 'status': 'OK', 'error': None}

    try:
        m, inter = run_core_pipeline(path, px_per_mm=PX_PER_MM)

        gray         = inter['gray']
        gear_cx      = m['gear_centre_x']
        gear_cy      = m['gear_centre_y']
        tip_r        = inter['tip_radius_px']
        root_r       = inter['root_radius_px']
        shaft_r      = inter['shaft_radius_px']
        h, w         = gray.shape

        print(f"Building gear mask...")
        gear_only, gear_filled = build_gear_mask(
            gray, gear_cx, gear_cy, shaft_r, tip_r)

        # ── Radial duty cycle scan on binary mask ──────────────────────────────────
        radii, duties = radial_scan_binary(
            gear_filled, gear_cx, gear_cy,
            start_radius=shaft_r + 5,
            end_radius=tip_r + 10)

        print(f"\nDuty cycle at key radii:")
        print(f"{'Radius':>10}  {'Duty':>8}  {'Fraction':>10}")
        print("─" * 35)
        for r, d in zip(radii[::10], duties[::10]):
            bar = "▓" * int(d * 20)
            print(f"{r:>10.1f}  {d:>8.3f}  {bar}")

        # Find 50% crossing
        pitch_r_binary = float(tip_r)
        for i in range(len(duties)-1):
            if duties[i] <= 0.5 <= duties[i+1]:
                frac           = (0.5 - duties[i]) / (duties[i+1] - duties[i])
                pitch_r_binary = float(radii[i] + frac)
                break

        print(f"\nTrue pitch radius (binary) : {pitch_r_binary:.1f} px")

        # ── Sample at three radii on binary mask ───────────────────────────────────
        sample_radii = {
            'root'       : root_r,
            'pitch'      : pitch_r_binary,
            'tip-5%'     : tip_r - 0.05 * (tip_r - root_r),
        }

        fig, axes = plt.subplots(2 + len(sample_radii), 1,
                                figsize=(16, 4 * (2 + len(sample_radii))))
        fig.suptitle("Binary mask analysis", fontsize=13)

        # Panel 1: gear mask
        vis = cv2.cvtColor(gear_only, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (gear_cx, gear_cy),
                int(pitch_r_binary), (0, 255, 0), 2)
        cv2.circle(vis, (gear_cx, gear_cy),
                int(tip_r), (0, 140, 255), 2)
        cv2.circle(vis, (gear_cx, gear_cy),
                int(root_r), (0, 0, 255), 2)
        axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Gear mask  "
                        "orange=tip  green=pitch  red=root")
        axes[0].axis('off')

        # Panel 2: duty cycle curve
        axes[1].plot(radii, duties, color='steelblue', linewidth=1)
        axes[1].axhline(0.5, color='orange', linestyle='--',
                        linewidth=1, label='50% duty')
        axes[1].axvline(pitch_r_binary, color='green',
                        linestyle='--', linewidth=1,
                        label=f'pitch r={pitch_r_binary:.0f}px')
        axes[1].axvline(root_r, color='red', linestyle=':', linewidth=1,
                        label=f'root r={root_r:.0f}px')
        axes[1].axvline(tip_r, color='orange', linestyle=':', linewidth=1,
                        label=f'tip r={tip_r:.0f}px')
        axes[1].set_title("Duty cycle vs radius (binary mask)")
        axes[1].set_xlabel("Radius (px)")
        axes[1].set_ylabel("Fraction white (tooth)")
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(0, 1)

        # Panels 3+: ring signals at sample radii
        colours = ['red', 'green', 'blue']
        for i, (label, r) in enumerate(sample_radii.items()):
            angles, signal = sample_ring_binary(
                gear_filled, gear_cx, gear_cy, r)
            ax = axes[2 + i]
            ax.plot(angles, signal, color=colours[i],
                    linewidth=0.5, alpha=0.8)
            ax.set_title(f"Binary signal at {label} (r={r:.0f}px)  "
                        f"duty={signal.mean():.3f}")
            ax.set_xlim(0, 360)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Angle (degrees)")
            ax.set_ylabel("Binary (0=gap, 1=tooth)")
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig("results/mask_analysis.png", dpi=150)
        plt.show()
    except Exception as e:
        result['status'] = 'ERROR'
        result['error']  = str(e)
        print(f"{fname:<20} ERROR: {e}")