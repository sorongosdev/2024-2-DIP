import numpy as np, cv2, time
from ch10.dft_1d import dft, idft


def calc_spectrum(complex):
    if complex.ndim == 2:
        dst = abs(complex)
    else:
        dst = cv2.magnitude(complex[:, :, 0], complex[:, :, 1])
    dst = 20 * np.log(dst + 1)
    return cv2.convertScaleAbs(dst)


def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h, w = dst.shpe[:2]
    cy, cx = h // 2, w // 2
    dst[h - cy:, w - cx:] = np.copy(img[0:cy, 0:cx])
    dst[0:cy, 0:cx] = np.copy(img[h - cy:, w - cx:])
    dst[0:cy, w - cx] = np.copy(img[h - cy:, 0:cx])
    dst[h - cy:, 0:cx] = np.copy(img[0:cy, w - cx:])
    return dst


def dft2(image):
    tmp = [dft(row) for row in image]
    dst = [dft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def idft2(image):
    tmp = [idft(row) for row in image]
    dst = [idft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


def ck_time(mode = 0):
    global stime
    if (mode == 0):
        stime = time.perf_counter()
    elif (mode == 1):
        etime = time.perf_counter()
        print("수행시간 = %.5f sec" % (etime - stime))


image = cv2.read('images/dft_240.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('Read image failed')

ck_time(0)
