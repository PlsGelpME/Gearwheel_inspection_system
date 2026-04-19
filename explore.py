import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

IMG_DIR = "images"

# Get all images
images = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
])

if len(images) == 0:
    print("No images found in images/ folder")
    sys.exit()

print(f"Found {len(images)} images:")
print()

# Show first 4 images with stats
show_count = min(4, len(images))
fig, axes  = plt.subplots(2, show_count, figsize=(5 * show_count, 10))
if show_count == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for i in range(show_count):
    path = os.path.join(IMG_DIR, images[i])
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"{images[i]}")
    print(f"  Size    : {gray.shape[1]} x {gray.shape[0]} px")
    print(f"  Mean    : {gray.mean():.1f}")
    print(f"  Std     : {gray.std():.1f}")
    print(f"  Min/Max : {gray.min()} / {gray.max()}")
    print()

    # Original image
    axes[0][i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0][i].set_title(f"{images[i]}\n"
                          f"{gray.shape[1]}x{gray.shape[0]}  "
                          f"mean={gray.mean():.0f}")
    axes[0][i].axis('off')

    # Histogram
    axes[1][i].hist(gray.ravel(), bins=256, range=(0, 255),
                    color='steelblue', edgecolor='none')
    axes[1][i].axvline(gray.mean(), color='red',
                       linestyle='--', linewidth=1,
                       label=f"mean={gray.mean():.0f}")
    axes[1][i].set_title("Histogram")
    axes[1][i].set_xlabel("Pixel value")
    axes[1][i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/explore_output.png", dpi=150)
plt.show()
print("Saved to results/explore_output.png")