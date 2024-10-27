import cv2
import numpy as np


def contrastEnh(src, a1, b1, a2, b2, a3, b3):
    width, height = src.shape

    for i in range(width):
        for j in range(height):
            if 0 <= src[i][j] <= 50:
                src[i][j] = a1 * src[i][j] + b1
            elif 50 <= src[i][j] <= 200:
                src[i][j] = a2 * src[i][j] + b2
            elif 200 <= src[i][j]:
                src[i][j] = a3 * src[i][j] + b3

    return src


image = cv2.imread('images/contrast.jpg', cv2.IMREAD_GRAYSCALE)

dst = contrastEnh(image, 1, 50, 0.5, 30, 0.5, 40)

cv2.imshow('conversion image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
