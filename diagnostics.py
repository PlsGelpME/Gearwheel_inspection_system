
import cv2
import matplotlib.pyplot as plt
import os

low_circ = ['images/end_face/Gear_1/endface39.png', 'images/end_face/Gear_1/endface100.png',
            'images/end_face/Gear_1/endface73.png',  'images/end_face/Gear_1/endface62.png', 'images/end_face/Gear_1/endface82.png']

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Low circularity images — visual inspection', fontsize=12)

for ax, fname in zip(axes, low_circ):
    path = os.path.join(fname)
    img  = cv2.imread(path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(fname, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('results/low_circularity.png', dpi=150)
plt.show()
