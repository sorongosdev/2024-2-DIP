import numpy as np, cv2
import math
import scipy.fftpack as sf


def cos(n, k, N):
    return math.cos((n + 1 / 2) * math.pi * k / N)


def C(k, N):
    return math.sqrt(1 / N) if k == 0 else math.sqrt(2 / N)


def dct(g):
    N = len(g)
    f = [C(k, N) * sum(g[n] * cos(n, k, N) for n in range(N)) for k in range(N)]
