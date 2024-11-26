import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread('images/desk_grayscale.jpg')

gamma = 0.4


# 감마 보정 함수
def gamma_correction(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


# 감마 보정 적용
image_gamma = gamma_correction(image, gamma)


def getGaussianMask(ksize, sigmaX, sigmaY):
    sigma = 0.3 * ((np.array(ksize) - 1.0) * 0.5 - 1.0) + 0.8  # 표준 편차
    if sigmaX <= 0: sigmaX = sigma[0]
    if sigmaY <= 0: sigmaY = sigma[1]

    u = np.array(ksize) // 2
    x = np.arange(-u[0], u[0] + 1, 1)
    y = np.arange(-u[1], u[1] + 1, 1)
    x, y = np.meshgrid(x, y)

    ratio = 1 / (sigmaX * sigmaY * 2 * np.pi)
    v1 = x ** 2 / (2 * sigmaX ** 2)
    v2 = y ** 2 / (2 * sigmaY ** 2)
    mask = ratio * np.exp(-(v1 + v2))
    return mask / np.sum(mask)


def apply_gaussian_blur(image, ksize=3):
    # 사용자 정의 가우시안 마스크 생성
    gaussian_mask = getGaussianMask(ksize, 0, 0)

    # 이미지에 가우시안 마스크 적용
    blurred_image = cv2.filter2D(image, -1, gaussian_mask)
    return blurred_image


# 감마 보정된 이미지에 가우시안 블러 적용
blurred_image = apply_gaussian_blur(image_gamma, (3, 3))

# 히스토그램 시각화
plt.figure()
plt.title("Gamma Corrected Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(image_gamma.ravel(), bins=256, range=[0, 256])
plt.show()

plt.figure()
plt.title("Blurred Gamma Corrected Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(blurred_image.ravel(), bins=256, range=[0, 256])
plt.show()

# 이미지 표시
cv2.imshow('Original Image', image)
cv2.imshow('Gamma Corrected Image', image_gamma)
cv2.imshow('Blurred Gamma Corrected Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')
