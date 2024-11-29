import numpy as np
import cv2


def zero_crossing(image):
    zc_image = np.zeros(image.shape, dtype=np.uint8)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = image[i - 1:i + 2, j - 1:j + 2]
            min_val = patch.min()
            max_val = patch.max()
            if min_val < 0 and max_val > 0:
                zc_image[i, j] = 255
    return zc_image


width = 256
height = 256
channels = 1

file_path1 = 'images/girl.raw'
file_path2 = 'images/elaine.raw'

image_data1 = np.fromfile(file_path1, dtype=np.uint8)
image_data2 = np.fromfile(file_path2, dtype=np.uint8)

image1 = image_data1.reshape((height, width, channels))
image2 = image_data2.reshape((height, width, channels))
if image1 is None: raise Exception("image1 영상파일 읽기 오류")
if image2 is None: raise Exception("image2 영상파일 읽기 오류")

ksize = (9, 9)
gaus1 = cv2.GaussianBlur(image1, ksize, 0)
gaus2 = cv2.GaussianBlur(image2, ksize, 0)

laplacian1 = cv2.Laplacian(gaus1, cv2.CV_64F)
laplacian2 = cv2.Laplacian(gaus2, cv2.CV_64F)

edges1 = zero_crossing(laplacian1)
edges2 = zero_crossing(laplacian2)

cv2.imshow("LoG: girl", edges1)
cv2.imshow("LoG: elaine", edges2)
cv2.waitKey(0)
cv2.destroyAllWindows()
