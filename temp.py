import cv2
import numpy as np

# Load image
img_path = '/home/anurag/Downloads/000001.jpg'
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found")

# Points
points = np.array([
    [1280, 273],
    [185, 574],
    [295, 535],
    [1280, 306],
    [185, 585],
    [1280, 377],
    [185, 615],
    [295, 629],
    [1280, 344],
    [185, 604],
])

# Draw numbered circles
for i, (x, y) in enumerate(points, start=1):
    x, y = int(x), int(y)

    # Circle
    cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

    # Number (slightly offset so it's readable)
    cv2.putText(
        img,
        str(i),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

# Save output
out_path = '/home/anurag/Downloads/numbered_points.jpg'
cv2.imwrite(out_path, img)

print(f"Saved to {out_path}")