import cv2

# 이미지 읽기
img = cv2.imread("images/opencv.png", cv2.IMREAD_COLOR)

# BGR을 HSV로 변환
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 녹색 범위 마스크 생성
greenScreen = cv2.inRange(converted, (60 - 10, 100, 100), (60 + 10, 255, 255))

# greenScreen을 원본 이미지와 비트 OR 연산
# 마스크를 3채널로 변환하여 원본 이미지와의 비트 OR 연산을 수행
greenScreen_colored = cv2.merge((greenScreen, greenScreen, greenScreen))
result_with_green = cv2.bitwise_or(img, greenScreen_colored)

# 결과 이미지 표시
cv2.imshow("Original Image", img)
cv2.imshow("Green Screen Mask", greenScreen)
cv2.imshow("Result with Green Screen OR", result_with_green)

cv2.waitKey(0)
cv2.destroyAllWindows()
