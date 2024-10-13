import numpy as np
import cv2


def draw_histo(hist, shape=(200, 255)):
    hist_img = np.full(shape, 255, np.uint8)  # 흰색 배경 생성
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)  # 정규화
    gap = hist_img.shape[1] / hist.shape[0]

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        roi = (x, 0, w, int(h.item()))  # .item()으로 스칼라 값으로 변환
        cv2.rectangle(hist_img, roi, 0, cv2.FILLED)  # 히스토그램 막대 그리기
    return cv2.flip(hist_img, 0)  # 세로 방향으로 뒤집기


# 이미지 읽기
image = cv2.imread('images/crayfish.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception('No image')

# 히스토그램 계산
bins, ranges = [256], [0, 256]
hist = cv2.calcHist([image], [0], None, bins, ranges)

# 누적 히스토그램 계산
accum_hist = np.zeros(hist.shape[:2], np.float32)
accum_hist[0] = hist[0]

for i in range(1, hist.shape[0]):
    accum_hist[i] = accum_hist[i - 1] + hist[i]  # 수정된 부분

# 누적 히스토그램 정규화
accum_hist = (accum_hist / np.sum(hist)) * 255

# LUT를 이용한 이미지 변환
dst1 = cv2.LUT(image, accum_hist.astype("uint8"))

# OpenCV의 히스토그램 평활화
dst2 = cv2.equalizeHist(image)

# 각 히스토그램 계산
hist1 = cv2.calcHist([dst1], [0], None, bins, ranges)  # 사용자 정의 방법
hist2 = cv2.calcHist([dst2], [0], None, bins, ranges)  # OpenCV 방법

# 히스토그램 이미지 생성
hist_img = draw_histo(hist)
hist_img1 = draw_histo(hist1)
hist_img2 = draw_histo(hist2)

# 결과 출력
cv2.imshow("Original Image", image)
cv2.imshow("Original Histogram", hist_img)
cv2.imshow("User Defined Equalized Image", dst1)
cv2.imshow("User Defined Histogram", hist_img1)
cv2.imshow("OpenCV Equalized Image", dst2)
cv2.imshow("OpenCV Histogram", hist_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
