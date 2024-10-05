import numpy as np
import cv2

# 색 선언
orange, black = (0, 165, 255), (0, 0, 0)
# 검은색 배경
image = np.full((300, 500, 3), black, np.uint8)
# 화면 중앙
center = (image.shape[1] // 2, image.shape[0] // 2)
# 원 그림
cv2.circle(image, center, 100, orange, 2)
# 그레이스케일로 변환
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 저장
cv2.imwrite("images/gray_img.jpg", gray_img)
# 창 표출
cv2.imshow("Draw circles", gray_img)
# 키 입력 대기
cv2.waitKey(0)
