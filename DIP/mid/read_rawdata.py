import cv2
import numpy as np

width = 512
height = 512
channels = 1

file_path = 'images/peppers_mixed.raw'

image_data = np.fromfile(file_path, dtype=np.uint8)
print(image_data.shape)
image = image_data.reshape((height, width, channels))
if image is None: raise Exception("영상파일 읽기 오류")

cv2.imshow("image", image)
cv2.waitKey(0)
