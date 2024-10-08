import numpy as np, cv2


def calc_histo(image, hsize, range = [0, 256]):
    hist = np.zeros((hsize, 1), np.float32)
