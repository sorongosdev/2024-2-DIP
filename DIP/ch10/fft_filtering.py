import numpy as np, cv2
from fft_2d import fft2, ifft2, calc_spectrum, fftshift


def FFT(image, mode = 2):
    if mode == 1:
        dft = fft2(image)
    elif mode == 2:
        dft = np.fft.fft2(image)
    elif mode == 3:
        dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)
    spectrum = calc_spectrum(dft)
    return dft, spectrum
