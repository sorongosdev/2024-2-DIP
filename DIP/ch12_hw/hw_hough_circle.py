import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/dish.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

edges = cv2.Canny(gray_blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = image.copy()

for contour in contours:
    if len(contour) >= 5:  # 타원을 맞추기 위해 최소 5개의 점이 필요
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)

plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Elliptical Hough Transform")
plt.axis('off')
plt.show()
