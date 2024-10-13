import cv2
import numpy as np


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


def search_value_idx(hist, bias = 0):
    for i in range(hist.shape[0]):
        idx = np.abs(bias - i)
        if hist[idx] > 0: return idx
    return -1
