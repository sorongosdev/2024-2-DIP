import cv2
import numpy as np

image = cv2.imread('images/Lenna.jpg')

# 인코딩 감마
# 1보다 크면 이미지가 뿌옇고, 대비가 낮아짐
# 작으면 이미지가 뚜렷하고, 대비가 높아짐

gamma_low = 0.4
gamma_high = 1.5


# 감마 보정 함수
def gamma_correction(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype = np.uint8)
    return cv2.LUT(image, lut)


image_gamma_low = gamma_correction(image, gamma_low)
image_gamma_high = gamma_correction(image, gamma_high)

cv2.imshow('Original', image)
cv2.imshow('Gamma0.4', image_gamma_low)
cv2.imshow('Gamma1.5', image_gamma_high)
cv2.waitKey(0)
cv2.destroyAllWindows()
