import cv2
import numpy as np
import random

src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("No image")

image = cv2.cvtColor(cv2.imread("images/img.png"), cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
k = 3
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
adaptive_thresh = cv2.adaptiveThreshold(segmented_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Thresholding", adaptive_thresh)

_, img_edge = cv2.threshold(adaptive_thresh, 128, 255, cv2.THRESH_BINARY_INV)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_edge)

colors = [np.array([0, 0, 0], dtype = np.uint8)]
for i in range(1, num_labels):
    colors.append(np.array([random.randint(0, 255) for _ in range(3)], dtype = np.uint8))

img_color = np.zeros((src.shape[0], src.shape[1], 3), dtype = np.uint8)
for y in range(src.shape[0]):
    for x in range(src.shape[1]):
        label = labels[y, x]
        img_color[y, x] = colors[label]

cv2.imshow("Labeled map", img_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
