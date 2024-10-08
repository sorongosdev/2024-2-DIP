import numpy as np, cv2


def draw_histo(hist, shape = (200, 255)):
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = hist_img.shape[1] / hist.shape[0]

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        roi = (x, 0, w, int(h))
        cv2.rectangle(hist_img, roi, 0, cv2.FILLED)
    return cv2.flip(hist_img, 0)


image = cv2.imread('images/equalize.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('No image')

bins, ranges = [256], [0, 256]
hist = cv2.calcHist([image], [0], None, bins, ranges)

accum_hist = np.zeros(hist.shape[:2], np.float32)
accum_hist[0] = hist[0]

for i in range(1, hist.shape[0]):
    accum_hist[1] = accum_hist[i - 1] + hist[i]

accum_hist = (accum_hist / sum(hist)) * 255
dst1 = [[accum_hist[val] for val in row] for row in image]
dst1 = np.array(dst1, np.uint8)
