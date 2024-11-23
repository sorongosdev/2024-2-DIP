import cv2
import numpy as np


# BGR에서 HSV로 변환
def bgr2hsv(bgr_img):
    # BGR -> HSV로 변환
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv_img


# HSV에서 각 채널을 계산
def cal_hsv(hsv_img):
    H = hsv_img[:, :, 0]  # Hue 채널 (0 ~ 179)
    S = hsv_img[:, :, 1]  # Saturation 채널 (0 ~ 255)
    V = hsv_img[:, :, 2]  # Value 채널 (0 ~ 255)

    return H, S, V


# 이미지 읽기
img = cv2.imread("images/opencv.png", cv2.IMREAD_COLOR)

# BGR -> HSV 변환
hsv_img = bgr2hsv(img)

# HSV 채널 추출
H, S, V = cal_hsv(hsv_img)

# Hue 범위 설정 (예: 빨간색 범위, 0 ~ 60)
H_mask = cv2.inRange(H, 0, 60)  # 빨간색 (Hue가 0 ~ 60 사이)

# 채도가 낮은 부분을 포함하려면 S가 낮을 때 (0 ~ 0.2)
S_mask = cv2.inRange(S, 0.0, 51)  # 낮은 채도 (0 ~ 51 범위)

# 밝기가 낮은 부분을 포함하려면 V가 낮을 때 (0 ~ 0.2)
V_mask = cv2.inRange(V, 0, 51)  # 밝기 낮은 부분 (0 ~ 51 범위)

# 마스크 반전
H_mask_inverted = cv2.bitwise_not(H_mask)
S_mask_inverted = cv2.bitwise_not(S_mask)
V_mask_inverted = cv2.bitwise_not(V_mask)

# 결과 이미지 표시
cv2.imshow("Hue channel", H_mask_inverted)
cv2.imshow("Saturation channel", S_mask_inverted)
cv2.imshow("Value channel", V_mask_inverted)

cv2.waitKey(0)
cv2.destroyAllWindows()
