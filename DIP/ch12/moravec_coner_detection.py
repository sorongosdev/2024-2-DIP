import cv2
import numpy as np
import matplotlib.pyplot as plt


def moravec_corner_detection(image, window_size = 3, threshold = 100):
    # image size
    height, width = image.shape
    offset = window_size // 2

    corner_response = np.zeros_like(image, dtype = np.float32)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            min_e = float('inf')
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                e = 0
                for wy in range(-offset, offset + 1):
                    for wx in range(-offset, offset + 1):
                        try:
                            i1 = image[y + wy, x + wx]
                            i2 = image[y + wy + dy, x + wx + dx]
                            e += (i1 - i2) ** 2
                        except IndexError:
                            pass
                min_e = min(min_e, e)
            corner_response[y, x] = min_e
    corners = np.argwhere(corner_response > threshold)
    return corners, corner_response


image = cv2.imread('images/checker.jpg', cv2.IMREAD_GRAYSCALE)

# detect MORAVEC corners
corners, corner_response = moravec_corner_detection(image, window_size = 3, threshold = 100)

# visualize
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for y, x in corners:
    cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Moravec Corner Detection")
plt.axis('off')
plt.show()

## complete
