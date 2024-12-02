import cv2
import numpy as np

# 이미지 읽기
src = cv2.imread("images/img.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("No image")

# 가우시안 블러 적용 (커널 사이즈 5)
blurred_img = cv2.GaussianBlur(src, (5, 5), 0)
cv2.imshow("Gaussian Blurred Image", blurred_img)

# Adaptive Thresholding 적용
adaptive_thresh = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Thresholding", adaptive_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
