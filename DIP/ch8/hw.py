import cv2
import numpy as np

# 이미지와 마스크 불러오기
original_image = cv2.imread('images/src.jpg')
mask1 = cv2.imread('images/fruit2.png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('images/fruit5.png', cv2.IMREAD_GRAYSCALE)

# 마스크를 이진화
_, binary_mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
_, binary_mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

# 원본 이미지와 마스크 간의 겹치는 부분 찾기
overlap_with_mask1 = cv2.bitwise_and(original_image, original_image, mask=binary_mask1)
overlap_with_mask2 = cv2.bitwise_and(original_image, original_image, mask=binary_mask2)

# 겹치는 픽셀 수 세기
overlap_count_mask1 = cv2.countNonZero(binary_mask1)
overlap_count_mask2 = cv2.countNonZero(binary_mask2)

# 결과 출력
print(f'mask1과 겹치는 픽셀 수: {overlap_count_mask1}')
print(f'mask2과 겹치는 픽셀 수: {overlap_count_mask2}')

# 결과 이미지 리사이즈
resize_width = 800  # 원하는 너비
resize_height1 = int((resize_width / original_image.shape[1]) * original_image.shape[0])  # 비율 유지
resize_height2 = int((resize_width / original_image.shape[1]) * original_image.shape[0])  # 비율 유지
overlap_with_mask1_resized = cv2.resize(overlap_with_mask1, (resize_width, resize_height1))
overlap_with_mask2_resized = cv2.resize(overlap_with_mask2, (resize_width, resize_height2))

# 결과 시각화
cv2.imshow('Overlap with Mask 1', overlap_with_mask1_resized)
cv2.imshow('Overlap with Mask 2', overlap_with_mask2_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
