import numpy as np, cv2
from fft_filtering import FFT, IFFT, calc_spectrum


def onRemoveMoire(val):
    radius = cv2.getTrackbarPos('Radius', title)
    th = cv2.getTrackbarPos('threshold', title)

    mask = cv2.threshold(spectrum_img, th, 255, cv2.THRESH_BINARY_INV)[1]
    y, x = np.divmod(mask.shpe, 2)[0]
    cv2.circle(mask,(x,y),radius,255,-1)

    if dft.ndim<3:
        remv_dft = np.zeros(dft.shape,np.complex)
        remv_dft.imag =
