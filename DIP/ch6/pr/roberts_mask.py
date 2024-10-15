import numpy as np, cv2
from blur import filter


def differential(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data2, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2)
    dst1, dst2 = np.abs(dst1), np.abs(dst2)

    dst = np.clip(dst)
