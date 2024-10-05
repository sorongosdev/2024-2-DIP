import numpy as np
import cv2

blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
image = np.zeros((400, 600, 3), np.uint8)  # 3채널 컬러영상
image[:] = (255, 255, 255)

pt1, pt2 = (50, 50), (250, 150)
pt3, pt4 = (400, 150), (500, 50)
roi = (50, 200, 200, 100)  # x, y, w, h

# 직선
cv2.line(image, pt1, pt2, red)
cv2.line(image, pt3, pt4, green, 3, cv2.LINE_AA)  # 계단 현상 감소선

cv2.rectangle(image, pt1, pt2, blue, 3, cv2.LINE_4)  # 4방향 연결선
cv2.rectangle(image, roi, red, 3, cv2.LINE_8)  # 8방향 연결선
cv2.rectangle(image, (400, 200, 100, 100), green, cv2.FILLED)  # 내부 채움

cv2.imshow("Line & Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
