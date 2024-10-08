import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./images/low2.jpg", cv2.IMREAD_GRAYSCALE)

_, global_thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

blur = cv2.GaussianBlur(image, (5, 5), 0)
_, otsu_thresh_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

mean_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

gaussian_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image', 'Global Thresholding', 'otsu thresholding', 'gaussian otsu thresholding',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [image, global_thresh, otsu_thresh, ]
