import cv2
import numpy as np

image = cv2.imread("images/beach.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3
_, labels, center = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(center)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

cv2.imshow("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
