import numpy as np, cv2


def median_filter(image, ksize):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            y1, y2 = i - center, i + center + 1
            x1, x2 = j - center, j + center + 1
            mask = image[y1:y2, x1:x2].flatten()
            sort_mask = cv2.sort(mask, cv2.SORT_EVERY_COLUMN).flat
            dst[i, j] = sort_mask[mask.size // 2]
    return dst

def salt_pepper_noise(img,n):

