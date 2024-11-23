import numpy as np, cv2
from ch10.dft_1d import exp


def zeropadding(img):
    h, w = img.shape[:2]
    m = 1 << int(np.ceil(np.log2(h)))
    n = 1 << int(np.ceil(np.log2(w)))
    dst = np.zeros((m, n), img.dtype)
    dst[0:h, 0:w] = img[:]
    return dst


def butterfly(pair, L, N, dir):
    for k in range(L):  # 버터플라이 수행
        Geven, Godd = pair[k], pair[k + L]
        pair[k] = Geven + Godd * exp(dir * k / N)  # 짝수부
        pair[k + L] = Geven - Godd * exp(dir * k / N)  # 홀수부


def pairing(g, N, dir, start=0, stride=1):
    if N == 1: return [g[start]]
    L = N // 2
    sd = stride * 2
    part1 = pairing(g, L, dir, start, sd)
    part2 = pairing(g, L, dir, start + stride, sd)
    pair = part1 + part2  # 결과 병합
    butterfly(pair, L, N, dir)
    return pair


def fft(g):
    return pairing(g, len(g), 1)


def ifft(g):
    fft = pairing(g, len(g), -1)
    return [v / len(g) for v in fft]


def fft2(image):
    pad_img = zeropadding(image)  # 영삽입
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)  # 전치 환원 후 반환


def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)  # 전치 환원 후 반환
