import numpy as np
import matplotlib.pyplot as plt


def exp(knN):
    th = -2 * math.pi * knN
    return complex(math.cos(th), math.sin(th))


def dft(g):
    N = len(g)
    dst = [sum(g[n] * exp(k * n / N) for n in range(N)) for k in range(N)]
    return np.array(dst)


def idft(G):
    N = len(G)
    dst = [sum(G[k] * exp(-k * n / N) for k in range(N)) for n in range(N)]
    return np.array(dst)


fmax = 1000
dt = 1 / fmax
t = np.arrange(0, 1, dt)

g1 = np.sin(2 * np.pi * 50 * t)
g2 = np.sin(2 * np.pi * 120 * t)
g3 = np.sin(2 * np.pi * 260 * t)
g1 = g1 * 0.6 + g2 * 0.9 + g3 * 0.2

N = len(g)
df = fmax
