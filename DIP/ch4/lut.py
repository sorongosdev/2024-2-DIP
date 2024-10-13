import cv2
import numpy as np

image = cv2.imread('images/Lenna.jpg', cv2.IMREAD_GRAYSCALE)

threshold = 127

lut = np.array([0 if i < threshold else 255 for i in range(256)], dtype=np.uint8)

binary_image = cv2.LUT(image, lut)

cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
