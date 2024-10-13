import cv2
import numpy as np

image = cv2.imread('images/Lenna.jpg', cv2.IMREAD_GRAYSCALE)

threshold1 = 50
threshold2 = 200

lut = np.array([
    np.clip(i + 50, 0, 255) if i < threshold1 else
    np.clip(0.5 * i + 30, 0, 255) if i < threshold2 else
    np.clip(0.5 * i + 40, 0, 255)
    for i in range(256)
], dtype=np.uint8)

new_image = cv2.LUT(image, lut)

cv2.imshow('Oringinal Image', image)
cv2.imshow('Linear Contrast Stretch', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
