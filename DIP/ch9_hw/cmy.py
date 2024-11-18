import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread("images/opencv.png", cv2.IMREAD_COLOR)

# BGR -> RGB 변환
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# RGB -> CMY 변환 (1 - RGB)
cmy = 1 - img_rgb / 255.0  # CMY 계산 (0 ~ 1 사이의 값으로 변환)

# CMY 채널 추출 (0 ~ 255 사이로 변환)
cyan_channel = (cmy[:, :, 0] * 255).astype(np.uint8)
magenta_channel = (cmy[:, :, 1] * 255).astype(np.uint8)
yellow_channel = (cmy[:, :, 2] * 255).astype(np.uint8)

# 각 CMY 채널을 색상 채널로 하여 출력
cyan_image = np.zeros_like(img)
cyan_image[:, :, 1] = cyan_channel  # Cyan에 해당하는 채널만 강조
cyan_image[:, :, 2] = cyan_channel  # Cyan에 해당하는 채널만 강조

magenta_image = np.zeros_like(img)
magenta_image[:, :, 0] = magenta_channel  # Magenta에 해당하는 채널만 강조
magenta_image[:, :, 2] = magenta_channel  # Magenta에 해당하는 채널만 강조

yellow_image = np.zeros_like(img)
yellow_image[:, :, 0] = yellow_channel  # Yellow에 해당하는 채널만 강조
yellow_image[:, :, 1] = yellow_channel  # Yellow에 해당하는 채널만 강조

# 결과 이미지 표시
cv2.imshow("Cyan Channel", cyan_image)
cv2.imshow("Magenta Channel", magenta_image)
cv2.imshow("Yellow Channel", yellow_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
