import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/building.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)

lines = cv2.HoughLines(edges, 1, rho = 1, theta = np.pi / 180, threshold = 150)

output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        alpha = 1000
        x1 = int(x0 + alpha * (-b))
        y1 = int(y0 + alpha * a)
        x2 = int(x0 - alpha * (-b))
        y2 = int(y0 - alpha * a)
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Hough Line Transform")
plt.axis('off')
plt.show()

# complete
