import numpy as np, cv2
from ch10.fft_2d import fft2, ifft2
from ch10.dft_2d import calc_spectrum, fftshift


def FFT(image, mode=2):
    if mode == 1:
        dft = fft2(image)
    elif mode == 2:
        dft = np.fft.fft2(image)
    elif mode == 3:
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)
    spectrum = calc_spectrum(dft)
    return dft, spectrum


def IFFT(dft, shape, mode=2):
    dft = fftshift(dft)  # 역 셔플링
    if mode == 1: img = ifft2(dft).real
    if mode == 2: img = np.fft.ifft2(dft).real
    if mode == 3:  img = cv2.idft(dft, flags=cv2.DFT_SCALE)[:, :, 0]
    img = img[:shape[0], :shape[1]]  # 영삽입 부분 제거
    return cv2.convertScaleAbs(img)
