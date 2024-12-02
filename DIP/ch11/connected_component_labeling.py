import cv2
import numpy as np
import random

img = cv2.imread("images/coins.png", cv2.IMREAD_GRAYSCALE)

_, img_edge = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Image after threshold", img_edge)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_edge)

colors = [np.array([0, 0, 0], dtype = np.uint8)]
for i in range(1, num_labels):
    colors.append(np.array([random.randint(0, 255) for _ in range(3)], dtype = np.uint8))

img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        label = labels[y, x]
        img_color[y, x] = colors[label]

cv2.imshow("Labeled map", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# to do.
