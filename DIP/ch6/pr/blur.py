import numpy as np, cv2, time


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


image = cv2.imread("images/filter_blur.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("No image")

date = [1 / 9, 1 / 9, 1 / 9,
        1 / 9, 1 / 9, 1 / 9,
        1 / 9, 1 / 9, 1 / 9]

mask = np.array(date, np.float32).reshape(3, 3)
blur1 = filter(image, mask)
# blur2 = filter2(image, mask)

cv2.imshow("image", image)
cv2.imshow("blur", blur1.astype("uint8"))
# cv2.imshow("blur", blur1.astype("uint8"))
cv2.waitKey(0)
