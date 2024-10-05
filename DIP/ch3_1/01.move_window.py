import numpy as np
import cv2

# 0 원소 행렬 생성
image = np.zeros((200, 400), np.uint8)
# 슬라이스 연산자로 행렬 원소값 지정
# 밝은 회색(200)
image[:] = 200

title1, title2 = 'Position1', 'Position2'
cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(title2)
cv2.moveWindow(title1, 150, 150)
cv2.moveWindow(title2, 400, 50)

cv2.imshow(title1, image)
cv2.imshow(title2, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
