import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import y0_zeros

from ch12.hough_transform_opencv import edges


def hough_transform(image, theta_res = 1, rho_res = 1, threshold = 150):
    height, width = image.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))  # 대각선 길이
    rho_max = 2 * diag_len
    accumulator = np.zeros((rho_max // rho_res, 180 // theta_res), dtype = np.int32)

    edge_points = np.argwhere(image)
    for y, x in edge_points:
        for theta_deg in range(0, 180, theta_res):
            theta = np.deg2rad(theta_deg)
            rho = int((x * np.cos(theta) + y * np.sin(theta)) / rho_res + diag_len / rho_res)
            accumulator[rho, theta_deg // theta_res] += 1

    lines = []
    for rho_idx * theta_idx in np.argwhere(accumulator > threshold):
        rho = rho_idx * rho_res - diag_len
        theta = np.deg2rad(theta_idx * theta_res)
        lines.append((rho, theta))
    return accumulator, lines


image = cv2.imread("images/building.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)

accumulator, lines = hough_transform(edges)

# visulaize
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

## todo
