import numpy as np
import cv2


def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    xcenter, ycenter = mask.shape[1] // 2, mask.shape[0] // 2

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = image[y1:y2, x1:x2].astype("float32")

            tmp = cv2.multiply(roi, mask)
            dst[i, j] = cv2.sumElems(tmp)[0]
    return dst


def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2)

    dst = cv2.convertScaleAbs(dst)
    dst1 = cv2.convertScaleAbs(dst1)
    dst2 = cv2.convertScaleAbs(dst2)
    return dst, dst1, dst2


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def apply_threshold(image, threshold):
    _, thresh_img = cv2.threshold(image, int(threshold * 255), 255, cv2.THRESH_BINARY)
    return thresh_img


def edge_thinning(sobel_edge, sobel_x, sobel_y):
    rows, cols = sobel_edge.shape
    thinned_edges = np.zeros_like(sobel_edge, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gx = sobel_x[i, j]
            gy = sobel_y[i, j]
            edge_strength = sobel_edge[i, j]

            if abs(gx) > abs(gy):  # Horizontal edge
                if edge_strength >= sobel_edge[i, j - 1] and edge_strength >= sobel_edge[i, j + 1]:
                    thinned_edges[i, j] = edge_strength
            else:  # Vertical edge
                if edge_strength >= sobel_edge[i - 1, j] and edge_strength >= sobel_edge[i + 1, j]:
                    thinned_edges[i, j] = edge_strength

    return thinned_edges


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

data1 = [-1, 0, 1,
         -2, 0, 2,
         -1, 0, 1]
data2 = [-1, -2, -1,
         0, 0, 0,
         1, 2, 1]

threshold = 0.4

# Process girl.raw
dst1, dst1_v, dst1_h = differential(image1, data1, data2)
dst1_sobel_v = cv2.Sobel(np.float32(image1), cv2.CV_32F, 1, 0, 3)
dst1_sobel_h = cv2.Sobel(np.float32(image1), cv2.CV_32F, 0, 1, 3)
dst1_sobel_v = cv2.convertScaleAbs(dst1_sobel_v)
dst1_sobel_h = cv2.convertScaleAbs(dst1_sobel_h)

dst1_v_norm = normalize(dst1_v)
dst1_h_norm = normalize(dst1_h)

thinned_edges1 = edge_thinning(dst1, dst1_sobel_v, dst1_sobel_h)

thinned_edges1_thresh = apply_threshold(thinned_edges1, threshold)

cv2.imshow("dst1: Thinned Edges", thinned_edges1)
cv2.imshow("dst1: Thinned Edges Thresh", thinned_edges1_thresh)

# Process elaine.raw
dst2, dst2_v, dst2_h = differential(image2, data1, data2)
dst2_sobel_v = cv2.Sobel(np.float32(image2), cv2.CV_32F, 1, 0, 3)
dst2_sobel_h = cv2.Sobel(np.float32(image2), cv2.CV_32F, 0, 1, 3)
dst2_sobel_v = cv2.convertScaleAbs(dst2_sobel_v)
dst2_sobel_h = cv2.convertScaleAbs(dst2_sobel_h)

dst2_v_norm = normalize(dst2_v)
dst2_h_norm = normalize(dst2_h)

thinned_edges2 = edge_thinning(dst2, dst2_sobel_v, dst2_sobel_h)

thinned_edges2_thresh = apply_threshold(thinned_edges2, threshold)

cv2.imshow("dst2: Thinned Edges", thinned_edges2)
cv2.imshow("dst2: Thinned Edges Thresh", thinned_edges2_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
