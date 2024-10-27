import cv2
import numpy as np

image = cv2.imread('images/gamma1.jpg', cv2.IMREAD_GRAYSCALE)

# 감마 값
# 1보다 크면 이미지가 뿌옇고, 대비가 낮아짐
# 작으면 이미지가 뚜렷하고, 대비가 높아짐

gamma = 0.55


# 감마 보정 함수
def gamma_correction(image, gamma):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


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


image_gamma = gamma_correction(image, gamma)

# 히스토그램 계산
bins, ranges = [256], [0, 256]
org_hist = cv2.calcHist([image], [0], None, bins, ranges)
gmm_hist = cv2.calcHist([image_gamma], [0], None, bins, ranges)

# 히스토그램 이미지 생성
org_hist_img = draw_histo(org_hist)
gmm_hist_img = draw_histo(gmm_hist)

cv2.imshow('Original', image)
cv2.imshow("Original Histogram", org_hist_img)

cv2.imshow('Gamma', image_gamma)
cv2.imshow('Gamma Histogram', gmm_hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
