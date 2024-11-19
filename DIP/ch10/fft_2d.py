import numpy as np, cv2
from ch10.dft_2d import exp, cal_spectrum, fftshift


def fft(g):
    return pairing(g, len(g), 1)


def ifft(g):
    fft = pairing(g, len(g), -1)


def fft2(image):
    pad_img = zeropadding(image)
    tmp = [fft(row) for row in pad_img]
    return np.transpose(dst)


def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)


image = cv2.imread('images/dft_240.jpg', cv2.IMREAD_GRAYSCALE)

dft1 = fft2(image)
dft2 = np.fft.fft2(image)
dft3 = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)

spectrum1 = calc_spectrum(fftshift(dft1))
spectrum2 = calc_spectrum(fftshift(dft2))
spectrum3 = calc_spectrum(fftshift(dft3))
