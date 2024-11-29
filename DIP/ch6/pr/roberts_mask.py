import numpy as np, cv2


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
    dst1, dst2 = np.abs(dst1), np.abs(dst2)

    dst = np.clip(dst, 0, 255).astype("uint8")
    dst1 = np.clip(dst1, 0, 255).astype("uint8")
    dst2 = np.clip(dst2, 0, 255).astype("uint8")
    return dst, dst1, dst2


image = cv2.imread("images/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("No image")

data1 = [-1, 0, 0,
         0, 1, 0,
         0, 0, 0]
data2 = [0, 0, -1,
         0, 1, 0,
         0, 0, 0]
dst, dst1, dst2 = differential(image, data1, data2)

cv2.imshow("image", image)
cv2.imshow("roberts edge", dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
