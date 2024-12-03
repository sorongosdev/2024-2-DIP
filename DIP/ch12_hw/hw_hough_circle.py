import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/dish.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 가우시안 블러로 노이즈 제거
circles = cv2.HoughCircles(gray_blurred,
                           cv2.HOUGH_GRADIENT,
                           1,
                           gray_blurred.shape[0] / 8,
                           200,
                           param2 = 50,
                           minRadius = 0,
                           maxRadius = 0)

output = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, radius = circle
        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Circular Hough Transform")
plt.axis('off')
plt.show()
