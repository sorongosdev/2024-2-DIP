import cv2
import numpy as np

width = 400
height = 300
channels = 3

file_path = 'images/desk.raw'

image_data = np.fromfile(file_path, dtype=np.uint8)
print(image_data.shape)
image = image_data.reshape((height, width, channels))
if image is None: raise Exception("영상파일 읽기 오류")

gray_image = 0.21 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 0]
gray_image = gray_image.astype(np.uint8)

cv2.imwrite('images/desk_grayscale.jpg', gray_image)

cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
