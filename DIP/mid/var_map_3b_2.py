import numpy as np
import cv2


def calculate_mean(window):
    return np.sum(window) / window.size


def calculate_variance(window, mean):
    return np.sum((window - mean) ** 2) / window.size


def variance_map(image):
    rows, cols = image.shape
    var_map = np.zeros((rows, cols), np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2]
            mean = calculate_mean(window)
            variance = calculate_variance(window, mean)
            var_map[i, j] = variance

    return var_map


def edge_detection_using_variance(image):
    var_map = variance_map(image)
    edges = np.zeros_like(var_map, dtype=np.uint8)
    threshold = np.mean(var_map) * 0.7

    edges[var_map > threshold] = 255
    return edges


width = 256
height = 256
channels = 1

file_path1 = 'images/girl.raw'
file_path2 = 'images/elaine.raw'

image_data1 = np.fromfile(file_path1, dtype=np.uint8)
image_data2 = np.fromfile(file_path2, dtype=np.uint8)

image1 = image_data1.reshape((height, width))
image2 = image_data2.reshape((height, width))
if image1 is None: raise Exception("image1 영상파일 읽기 오류")
if image2 is None: raise Exception("image2 영상파일 읽기 오류")

edges1 = edge_detection_using_variance(image1)
edges2 = edge_detection_using_variance(image2)

cv2.imshow("Variance Map1", variance_map(image1).astype("uint8"))
cv2.imshow("Variance Map2", variance_map(image2).astype("uint8"))
cv2.imshow("Edges Detected1", edges1)
cv2.imshow("Edges Detected2", edges2)
cv2.waitKey(0)
cv2.destroyAllWindows()
