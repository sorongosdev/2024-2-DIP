import cv2

# 이미지 읽기
img = cv2.imread("images/img.png")

# 그레이스케일 변환
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_img)

# 가우시안 블러 적용 (커널 사이즈 5)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
cv2.imshow("Gaussian Blurred Image", blurred_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
