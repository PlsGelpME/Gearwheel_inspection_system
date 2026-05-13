import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.gear_core import run_core_pipeline

path      = "images/real_time_gear/rtg6.jpg"
PX_PER_MM = 20.0

m, inter  = run_core_pipeline(path, px_per_mm=PX_PER_MM)

gray         = inter['gray']
gear_cx      = m['gear_centre_x']
gear_cy      = m['gear_centre_y']
pitch_r      = inter['pitch_radius_px']
tip_r        = inter['tip_radius_px']
root_r       = inter['root_radius_px']
h, w         = gray.shape

# ── Sample at three radii to see which gives cleanest signal ───────────────
# We try pitch, and two intermediate radii between root and tip
sample_radii = {
    'root'          : root_r,
    'root+25%depth' : root_r + (tip_r - root_r) * 0.25,
    'pitch'         : pitch_r,
    'tip-10%depth'  : tip_r  - (tip_r - root_r) * 0.10,
}

NUM_ANGLES = 1440   # 0.25° resolution for clear signal
angles     = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
angle_degs = np.degrees(angles)

signals = {}
for label, r in sample_radii.items():
    xs = np.clip((gear_cx + r * np.cos(angles)).astype(int), 0, w-1)
    ys = np.clip((gear_cy + r * np.sin(angles)).astype(int), 0, h-1)
    signals[label] = gray[ys, xs].astype(float)

# ── Plot all four signals ──────────────────────────────────────────────────
fig, axes = plt.subplots(len(signals) + 1, 1,
                          figsize=(16, 3 * (len(signals) + 1)))
fig.suptitle("Brightness signal at different sampling radii\n"
             "looking for clearest tooth / gap alternation", fontsize=12)

colours = ['red', 'orange', 'green', 'blue']
for i, (label, signal) in enumerate(signals.items()):
    ax = axes[i]
    ax.plot(angle_degs, signal,
            color=colours[i], linewidth=0.6, alpha=0.8)
    ax.axhline(signal.mean(), color='gray', linestyle='--',
               linewidth=0.8, label=f'mean={signal.mean():.1f}')
    ax.set_title(f"Radius: {label}  (r={sample_radii[label]:.0f}px)")
    ax.set_xlim(0, 360)
    ax.set_ylabel("Brightness")
    ax.legend(fontsize=8)
    if i == len(signals) - 1:
        ax.set_xlabel("Angle (degrees)")

# ── Bottom panel: zoom into first 30° of pitch signal ─────────────────────
pitch_signal = signals['pitch']
zoom_mask    = angle_degs <= 30
axes[-1].plot(angle_degs[zoom_mask], pitch_signal[zoom_mask],
              color='green', linewidth=1.5)
axes[-1].axhline(pitch_signal.mean(), color='gray',
                 linestyle='--', linewidth=0.8)
axes[-1].set_title("Pitch circle signal — zoomed to first 30°\n"
                   "(each bright peak = one tooth, each dip = one gap)")
axes[-1].set_xlabel("Angle (degrees)")
axes[-1].set_ylabel("Brightness")
axes[-1].set_xlim(0, 30)

plt.tight_layout()
plt.savefig("results/tooth_signal.png", dpi=150)
plt.show()

# ── Print signal statistics ────────────────────────────────────────────────
print(f"Gear centre     : ({gear_cx}, {gear_cy})")
print(f"Pitch radius    : {pitch_r:.1f} px")
print(f"Tip radius      : {tip_r:.1f} px")
print(f"Root radius     : {root_r:.1f} px")
print()
for label, signal in signals.items():
    print(f"{label:20s}  min={signal.min():5.1f}  "
          f"max={signal.max():5.1f}  "
          f"mean={signal.mean():5.1f}  "
          f"std={signal.std():5.1f}")