import numpy as np
import cv2


def draw_histogram(hist_r, hist_g, hist_b, shape = (200, 256, 3)):
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist_r, hist_r, 0, shape[0], cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, 0, shape[0], cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, shape[0], cv2.NORM_MINMAX)

    gap = shape[1] // hist_r.shape[0]

    for i in range(hist_r.shape[0]):
        x = i * gap
        cv2.rectangle(hist_img, (x, shape[0]), (x + gap, shape[0] - int(hist_r[i].item())), (0, 0, 255), cv2.FILLED)
