import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images/desk_grayscale.jpg', cv2.IMREAD_GRAYSCALE)

threshold1 = 10
threshold2 = 250

lut = np.array([
    np.clip(1.2 * i + 20, 0, 255) if i < threshold1 else
    np.clip(0.6 * i + 25, 0, 255) if i < threshold2 else
    np.clip(0.5 * i + 50, 0, 255)
    for i in range(256)
], dtype=np.uint8)

new_image = cv2.LUT(image, lut)

plt.figure()
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(image.ravel(), bins=256, range=[0, 256])
plt.show()

plt.figure()
plt.title("Contrast Stretched Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.hist(new_image.ravel(), bins=256, range=[0, 256])
plt.show()

cv2.imshow('Original Image', image)
cv2.imshow('Linear Contrast Stretch', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close('all')
