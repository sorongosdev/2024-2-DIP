import numpy as np
import cv2

image1 = np.zeros((50, 512), np.uint8)
image2 = np.zeros((50, 512), np.uint8)

rows, cols = image1.shape[:2]

for i in range(rows):
    for j in range(cols):
        image1[i, j] = j // 2
        image2[i, j] = j // 20 * 10

cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.waitKey(0)
