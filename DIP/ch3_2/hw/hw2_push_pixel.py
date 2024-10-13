import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread("images/Lenna.jpg")

# 이동할 거리
ty = 40

# 전체 이미지 이동
translated_img = np.zeros_like(image)
translated_img[:-ty, :, :] = image[ty:, :, :]  # y,x 순

# 텍스트
cv2.putText(translated_img, "2020008768", (10, 390), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)

# 결과
cv2.imshow("Translated Image", translated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
