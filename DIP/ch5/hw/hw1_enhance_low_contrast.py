import numpy as np
import cv2

# 초기값 설정
norm_const = 255  # 정규화 범위


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


def update(val):
    global norm_const
    norm_const = cv2.getTrackbarPos('Norm Const', 'Result')
    process_image()


def process_image():
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 원본 이미지 히스토그램 계산
    bins, ranges = [256], [0, 256]
    hist_original = cv2.calcHist([gray_image], [0], None, bins, ranges)

    # 사용자 정의 누적 히스토그램 계산
    hist = cv2.calcHist([gray_image], [0], None, bins, ranges)

    # 누적 히스토그램 계산
    accum_hist = np.zeros(hist.shape[:2], np.float32)
    accum_hist[0] = hist[0]

    for i in range(1, hist.shape[0]):
        accum_hist[i] = accum_hist[i - 1] + hist[i]

    # 누적 히스토그램 정규화
    accum_hist = (accum_hist / np.sum(hist)) * norm_const  # 정규화 범위 조정

    # LUT를 이용한 사용자 정의 이미지 변환
    equalized_image = cv2.LUT(gray_image, accum_hist.astype("uint8"))

    # 각 히스토그램 계산 (평활화된 이미지에 대해)
    hist_user = cv2.calcHist([equalized_image], [0], None, bins, ranges)  # 사용자 정의 방법

    # 히스토그램 이미지 생성
    hist_img_original = draw_histo(hist_original)  # 원본 이미지 히스토그램
    hist_img_user = draw_histo(hist_user)  # 사용자 정의 히스토그램

    # 결과 출력
    cv2.imshow("Original Image (Grayscale)", gray_image)  # 원본 이미지를 그레이스케일로 표시
    cv2.imshow("Original Histogram", hist_img_original)  # 원본 이미지 히스토그램
    cv2.imshow("User Defined Equalized Image", equalized_image)  # 평활화된 이미지
    cv2.imshow("User Defined Histogram", hist_img_user)  # 사용자 정의 히스토그램


# 이미지 읽기
image = cv2.imread('images/crayfish.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('No image')

# 트랙바 생성
cv2.namedWindow('Result')
cv2.createTrackbar('Norm Const', 'Result', 0, 255, update)  # norm_const 조정

# 초기 이미지 처리
process_image()

cv2.waitKey(0)
cv2.destroyAllWindows()
