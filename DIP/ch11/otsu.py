import cv2
import numpy as np

src = cv2.imread("images/lenna.jpg", cv2.IMREAD_GRAYSCALE)
if src is None: print("no image")

_, binary_global = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)

_, binary_otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

otsu_threshold_value = _
print(f"Otsu's threshold value: {otsu_threshold_value}")

cv2.imread("Oringinal", src)
cv2.imshow("Global Threshold (128)", binary_global)
cv2.imshow("Otsu's Threshold", binary_otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
