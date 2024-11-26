import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/desk_grayscale.jpg')

gamma = 0.6


# 감마 보정 함수
def gamma_correction(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


image_gamma = gamma_correction(image, gamma)

plt.figure()
plt.title("Gamma Corrected Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(image_gamma.ravel(), bins=256, range=[0, 256])
plt.show()

cv2.imshow('Original', image)
cv2.imshow('Gamma Corrected', image_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')
