import cv2
import numpy as np


def median_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2].flatten()
            sort_mask = np.sort(mask)
            dst[i, j] = sort_mask[mask.size // 2]
    return dst


def getGaussianMask(ksize, sigmaX, sigmaY):
    sigma = 0.3 * ((np.array(ksize) - 1.0) * 0.5 - 1.0) + 0.8
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


width = 512
height = 512
channels = 1
file_path = 'images/peppers_mixed.raw'

image_data = np.fromfile(file_path, dtype=np.uint8)
print(image_data.shape)

image = image_data.reshape((height, width, channels))

if image is None:
    raise Exception("영상파일 읽기 오류")

med_img1 = median_filter(image, 3)

ksize = (5, 5)
gaussian_2d = getGaussianMask(ksize, 1, 1)

gauss_img1 = cv2.filter2D(med_img1, -1, gaussian_2d)

cv2.imshow("Original Image", image)
cv2.imshow("Median Filtered Image", med_img1)
cv2.imshow("Gaussian Smoothed Image", gauss_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
