import numpy as np
import cv2

# 이미지를 그레이스케일로 읽기
lenna_img = cv2.imread("images/Lenna.jpg", cv2.IMREAD_GRAYSCALE)

# 원본 이미지 크기 확인
h, w = lenna_img.shape[:2]
mask_img = np.zeros((h, w), np.uint8)

# 마스크 사각형 생성 (왼쪽 절반은 흰색, 오른쪽 절반은 검은색)
cx = w // 2
cv2.rectangle(mask_img, (0, 0), (cx, h), 255, -1)  # 왼쪽 절반을 흰색으로

and_img = cv2.bitwise_and(lenna_img, mask_img)
or_img = cv2.bitwise_or(lenna_img, mask_img)
xor_img = cv2.bitwise_xor(lenna_img, mask_img)

# 결과 보여주기
cv2.imshow("AND", and_img)
cv2.imshow("OR", or_img)
cv2.imshow("XOR", xor_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
