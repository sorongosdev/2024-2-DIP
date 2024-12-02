import cv2
import numpy as np

src = cv2.imread("images/lenna.jpg", cv2.IMREAD_GRAYSCALE)
if src is None: print("No image")

adaptive_mean = cv2.adaptiveThreshold(src, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C,
                                      thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)

adaptive_gaussian = cv2.adaptiveThreshold(src, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)

cv2.imshow("Original", src)
cv2.imshow("Adaptive Mean Thresholding", adaptive_mean)
cv2.imshow("Adaptive Gaussian Thresholding", adaptive_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()
