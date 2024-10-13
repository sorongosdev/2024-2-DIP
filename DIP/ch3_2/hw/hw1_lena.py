import cv2
import numpy as np

image = cv2.imread("images/Lenna.jpg")

# 채널
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# 채널 -30
blue_channel = np.clip(blue_channel - 30, 0, 255)  # 0보다 작지 않게 조정
green_channel = np.clip(green_channel - 30, 0, 255)
red_channel = np.clip(red_channel - 30, 0, 255)

# 수정된 채널을 원본 이미지에 적용
image_blue = image.copy()
image_green = image.copy()
image_red = image.copy()

image_blue[:, :, 0] = blue_channel
image_green[:, :, 1] = green_channel
image_red[:, :, 2] = red_channel

# 결과
cv2.imshow("Original Lenna", image)
cv2.imshow("Blue Channel Decreased", image_blue)
cv2.imshow("Green Channel Decreased", image_green)
cv2.imshow("Red Channel Decreased", image_red)

cv2.waitKey(0)
cv2.destroyAllWindows()
