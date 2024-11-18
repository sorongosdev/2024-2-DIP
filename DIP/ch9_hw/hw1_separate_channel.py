import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread("images/opencv.png", cv2.IMREAD_COLOR)

# BGR을 HSV로 변환
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 녹색 범위 마스크 생성 (green)
green_screen = cv2.inRange(converted, (60 - 20, 100, 100), (60 + 20, 255, 255))

# 파란색 범위 마스크 생성 (blue)
blue_screen = cv2.inRange(converted, (120 - 20, 100, 100), (120 + 20, 255, 255))

# 빨간색 범위 마스크 생성 (red)
red_screen = cv2.inRange(converted, (0 - 10, 100, 100), (10 + 10, 255, 255))

# 하얀 배경을 위한 마스크 생성 (배경이 완전히 흰색인 경우, BGR=(255,255,255)만을 선택)
white_screen = cv2.inRange(img, (255, 255, 255), (255, 255, 255))

# green_screen, blue_screen, red_screen 마스크 반전
green_screen_inverted = cv2.bitwise_not(green_screen)
blue_screen_inverted = cv2.bitwise_not(blue_screen)
red_screen_inverted = cv2.bitwise_not(red_screen)

# green_screen, blue_screen, red_screen과 white_screen의 XOR 연산
xor_green = cv2.bitwise_xor(green_screen, white_screen)
xor_blue = cv2.bitwise_xor(blue_screen, white_screen)
xor_red = cv2.bitwise_xor(red_screen, white_screen)

cv2.imshow("Green Channel", xor_green)
cv2.imshow("Blue Channel", xor_blue)
cv2.imshow("Red Channel", xor_red)
cv2.waitKey(0)
cv2.destroyAllWindows()
