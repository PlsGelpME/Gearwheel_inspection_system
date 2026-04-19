
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "V:\gearwheel_inspection3\images\WhatsApp Image 2026-04-18 at 12.10.00 (1).jpeg"  # use the _resized version
img  = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

blurred = cv2.GaussianBlur(gray, (5,5), sigmaX=0)
edges   = cv2.Canny(blurred, 64, 128)

# Draw horizontal lines at detected y positions
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Draw y lines at cluster positions
for y, col in [(309, (255,0,0)),    # blue  = isolated line
               (322, (0,255,0)),    # green = top cluster
               (462, (0,0,255)),    # red   = bottom cluster
               (556, (255,165,0))]: # orange= third cluster
    cv2.line(vis, (0,y), (w,y), col, 2)
    cv2.putText(vis, f'y={y}', (10, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title('Cluster y positions on image')
plt.axis('off')
plt.savefig('results/cluster_debug.png', dpi=150)
plt.show()
print('Saved to results/cluster_debug.png')
