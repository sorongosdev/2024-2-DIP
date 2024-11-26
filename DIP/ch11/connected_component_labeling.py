import cv2
import numpy as np
import random

img = cv2.imread("images/coins.png", cv2.IMREAD_GRAYSCALE)

_, img_edge = cv2.threshold("img", 128, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Image after threshold", img_edge)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_edge)

colors = [np.array([0, 0, 0], dtype = np.uint8)]
for i in range(1, num_labels):
    colors.append(np.array([random.randint(0, 255)]))

# to do.
