import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipeline.sideProfile import (run_sideprofile_pipeline,
                                   print_sideprofile_summary)

# ── Resize image to standard width before processing ──────────────────────
def load_and_resize(path, target_width=800):
    """Resize image to target width, preserve aspect ratio."""
    img  = cv2.imread(path)
    h, w = img.shape[:2]
    if w > target_width:
        scale  = target_width / w
        new_w  = target_width
        new_h  = int(h * scale)
        img    = cv2.resize(img, (new_w, new_h))
        print(f"Resized: {w}x{h} → {new_w}x{new_h}  (scale={scale:.3f})")
    # Save resized version temporarily
    resized_path = path.replace('.png', '_resized.png').replace(
                   '.jpg', '_resized.jpg')
    cv2.imwrite(resized_path, img)
    return resized_path

path      = "V:\gearwheel_inspection3\images\WhatsApp Image 2026-04-18 at 12.10.00 (1).jpeg"   # ← replace
path      = load_and_resize(path, target_width=800)
PX_PER_MM = 20.0
GEAR_TYPE = "helical"

print("Running side profile pipeline...")
print()

m, debug = run_sideprofile_pipeline(path, GEAR_TYPE, PX_PER_MM)
print_sideprofile_summary(m)

gray  = debug['gray']
face  = debug['face']
shaft = debug['shaft']
helix = debug['helix']
h, w  = gray.shape

# ── Build annotated image ──────────────────────────────────────────────────
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw top and bottom face edge lines
x_arr   = np.array([face['x_start'], face['x_end']])
y_top_l = (face['m_top'] * x_arr + face['c_top']).astype(int)
y_bot_l = (face['m_bot'] * x_arr + face['c_bot']).astype(int)

cv2.line(vis, (x_arr[0], y_top_l[0]),
         (x_arr[1], y_top_l[1]), (0, 255, 0), 2)   # green = top
cv2.line(vis, (x_arr[0], y_bot_l[0]),
         (x_arr[1], y_bot_l[1]), (0, 0, 255), 2)   # red   = bottom

# Face width arrow
x_mid = w // 2
y_t   = int(face['m_top'] * x_mid + face['c_top'])
y_b   = int(face['m_bot'] * x_mid + face['c_bot'])
cv2.arrowedLine(vis, (x_mid, y_t), (x_mid, y_b), (0,255,255), 2)
cv2.arrowedLine(vis, (x_mid, y_b), (x_mid, y_t), (0,255,255), 2)
cv2.putText(vis, f"{m['face_width_mm']:.1f}mm",
            (x_mid + 10, (y_t+y_b)//2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

# Shaft lines
if shaft:
    for c in shaft['cluster']:
        cv2.line(vis, (c['x1'], c['y1']),
                 (c['x2'], c['y2']), (255,255,0), 1)
    cv2.line(vis,
             (int(shaft['shaft_x_left']),  0),
             (int(shaft['shaft_x_left']),  y_t),
             (255, 0, 255), 2)
    cv2.line(vis,
             (int(shaft['shaft_x_right']), 0),
             (int(shaft['shaft_x_right']), y_t),
             (255, 0, 255), 2)

# Helix lines on gear face
if helix.get('diag_lines'):
    roi_top = helix['roi_top']
    x_off   = helix['x_start']
    for x1,y1,x2,y2 in helix['diag_lines']:
        cv2.line(vis,
                 (x1 + x_off, y1 + roi_top),
                 (x2 + x_off, y2 + roi_top),
                 (255, 165, 0), 1)   # orange = helix lines

# Legend
font  = cv2.FONT_HERSHEY_SIMPLEX
items = [
    ((0,   255, 0  ), "Top face edge"),
    ((0,   0,   255), "Bottom face edge"),
    ((0,   255, 255), f"Face width {m['face_width_mm']:.1f}mm"),
    ((255, 255, 0  ), "Shaft lines"),
    ((255, 0,   255), "Shaft boundary"),
    ((255, 165, 0  ), f"Helix lines ~{m['helix_angle_deg']:.1f}°"
                      if m['helix_angle_deg'] else "Helix: not detected"),
]
for i, (col, label) in enumerate(items):
    y = 25 + i * 24
    cv2.rectangle(vis, (8, y-12), (24, y+4), col, -1)
    cv2.putText(vis, label, (30, y), font, 0.5, (0,0,0),       2)
    cv2.putText(vis, label, (30, y), font, 0.5, (255,255,255), 1)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Side profile measurements", fontsize=13)

axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Annotated\n"
                   f"face={m['face_width_mm']:.1f}mm  "
                   f"helix={m['helix_angle_deg']:.1f}°"
                   if m['helix_angle_deg'] else "Annotated")
axes[0].axis('off')

# Angle distribution of helix lines
if helix.get('diag_lines'):
    all_angles = []
    for x1,y1,x2,y2 in helix['diag_lines']:
        a = np.degrees(np.arctan2(y2-y1, x2-x1))
        all_angles.append(a)
    axes[1].hist(all_angles, bins=30, color='steelblue',
                 edgecolor='white', linewidth=0.5)
    axes[1].axvline(helix['hough_angle_deg'], color='red',
                    linestyle='--',
                    label=f"mean={helix['hough_angle_deg']:.1f}°")
    axes[1].set_title(f"Helix line angle distribution\n"
                       f"helix angle = 90° - "
                       f"{abs(helix['hough_angle_deg']):.1f}° = "
                       f"{m['helix_angle_deg']:.1f}°")
    axes[1].set_xlabel("Line angle from horizontal (°)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
else:
    axes[1].set_title("No helix lines detected")
    axes[1].axis('off')

plt.tight_layout()
plt.savefig("results/sideprofile_output.png", dpi=150)
plt.show()