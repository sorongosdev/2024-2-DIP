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
            sort_mask = cv2.sort(mask, cv2.SORT_EVERY_COLUMN).flat
            dst[i, j] = sort_mask[mask.size // 2]
    return dst


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

cv2.imshow("image", image)
cv2.imshow("med_img1", med_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()