import numpy as np
import cv2
import matplotlib.pyplot as plt

norm_const = 100


def draw_histo(hist, shape=(200, 255)):
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = hist_img.shape[1] / hist.shape[0]

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        roi = (x, 0, w, int(h.item()))
        cv2.rectangle(hist_img, roi, 0, cv2.FILLED)
    return cv2.flip(hist_img, 0)


def process_image():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bins, ranges = [256], [0, 256]
    hist_original = cv2.calcHist([gray_image], [0], None, bins, ranges)
    hist = cv2.calcHist([gray_image], [0], None, bins, ranges)
    accum_hist = np.zeros(hist.shape[:2], np.float32)
    accum_hist[0] = hist[0]

    for i in range(1, hist.shape[0]):
        accum_hist[i] = accum_hist[i - 1] + hist[i]

    accum_hist = (accum_hist / np.sum(hist)) * norm_const
    equalized_image = cv2.LUT(gray_image, accum_hist.astype("uint8"))
    hist_user = cv2.calcHist([equalized_image], [0], None, bins, ranges)

    plt.figure()
    plt.title("Original Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.hist(gray_image.ravel(), bins=256, range=[0, 256])
    plt.show()

    plt.figure()
    plt.title("Histogram Equalized Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.hist(equalized_image.ravel(), bins=256, range=[0, 256])
    plt.show()

    cv2.imshow("Original Image (Grayscale)", gray_image)
    cv2.imshow("Histogram Equalized Image", equalized_image)


image = cv2.imread('images/desk_grayscale.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('No image')

process_image()

cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')
