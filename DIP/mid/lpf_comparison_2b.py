import numpy as np
import cv2


def average_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            if y1 < 0 or y2 > rows or x1 < 0 or x2 > cols:
                dst[i, j] = image[i, j]
            else:
                mask = image[y1:y2, x1:x2]
                dst[i, j] = cv2.mean(mask)[0]
    return dst


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


width = 512
height = 512
channels = 1
file_path = 'images/peppers_mixed.raw'

image_data = np.fromfile(file_path, dtype=np.uint8)
print(image_data.shape)

image = image_data.reshape((height, width, channels))

if image is None:
    raise Exception("영상파일 읽기 오류")
# 평균 필터 적용 (3x3 및 5x5)
avg_img_3x3 = average_filter(image, 3)
avg_img_5x5 = average_filter(image, 5)

# 가우시안 필터 적용 (3x3 및 5x5)
gaussian_mask_3x3 = getGaussianMask((3, 3), 0, 0)
gaussian_mask_5x5 = getGaussianMask((5, 5), 0, 0)
gauss_img_3x3 = cv2.filter2D(image, -1, gaussian_mask_3x3)
gauss_img_5x5 = cv2.filter2D(image, -1, gaussian_mask_5x5)

# 결과 표시
titles = ['Original Image', 'Avg Filter 3x3', 'Avg Filter 5x5', 'Gaussian Filter 3x3', 'Gaussian Filter 5x5']
images = [image, avg_img_3x3, avg_img_5x5, gauss_img_3x3, gauss_img_5x5]

for title, img in zip(titles, images):
    cv2.imshow(title, img)

cv2.waitKey(0)
cv2.destroyAllWindows()
