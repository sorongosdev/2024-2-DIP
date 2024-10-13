import numpy as np
import cv2

# 초기값 설정
norm_const = 0  # 정규화 범위
shift_value = 0  # 조정 밝기


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


def shift_brightness(image, shift_value):
    # 픽셀 값에 shift_value를 더함
    shifted_image = cv2.add(image, (shift_value, shift_value, shift_value, 0))
    return shifted_image


def update(val):
    # 트랙바에서 변경된 값에 따라 이미지를 업데이트
    global norm_const, shift_value
    norm_const = cv2.getTrackbarPos('Norm Const', 'Result')
    shift_value = cv2.getTrackbarPos('Shift Value', 'Result')

    # 이미지와 히스토그램 업데이트
    process_image()


def process_image():
    # 각 채널에 대해 사용자 정의 히스토그램 평활화
    channels = cv2.split(image)  # B, G, R 채널 분리
    equalized_channels_user = []

    # 원본 이미지 히스토그램 계산
    bins, ranges = [256], [0, 256]
    hist_original = cv2.calcHist([image], [0], None, bins, ranges)

    for channel in channels:
        # 사용자 정의 누적 히스토그램 계산
        hist = cv2.calcHist([channel], [0], None, bins, ranges)

        # 누적 히스토그램 계산
        accum_hist = np.zeros(hist.shape[:2], np.float32)
        accum_hist[0] = hist[0]

        for i in range(1, hist.shape[0]):
            accum_hist[i] = accum_hist[i - 1] + hist[i]

        # 누적 히스토그램 정규화
        accum_hist = (accum_hist / np.sum(hist)) * norm_const  # 정규화 범위 조정

        # LUT를 이용한 사용자 정의 이미지 변환
        equalized_channel_user = cv2.LUT(channel, accum_hist.astype("uint8"))
        equalized_channels_user.append(equalized_channel_user)  # 평활화된 채널 저장

    # 평활화된 채널을 다시 합치기
    dst_user = cv2.merge(equalized_channels_user)

    # 밝기 조정
    dst_user = shift_brightness(dst_user, shift_value)  # 평활화된 이미지에 밝기 조정 적용

    # 각 히스토그램 계산 (밝기 조정 후 이미지에 대해 계산)
    hist_user = cv2.calcHist([dst_user], [0], None, bins, ranges)  # 사용자 정의 방법

    # 히스토그램 이미지 생성
    hist_img_original = draw_histo(hist_original)  # 원본 이미지 히스토그램
    hist_img_user = draw_histo(hist_user)  # 사용자 정의 평활화 히스토그램

    # 결과 출력
    cv2.imshow("Original Image", image)
    cv2.imshow("Original Histogram", hist_img_original)
    cv2.imshow("User Defined Equalized Image", dst_user)
    cv2.imshow("User Defined Histogram", hist_img_user)


# 이미지 읽기
image = cv2.imread('images/crayfish.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('No image')

# 트랙바 생성
cv2.namedWindow('Result')
cv2.createTrackbar('Norm Const', 'Result', 0, 255, update)  # norm_const 조정
cv2.createTrackbar('Shift Value', 'Result', 0, 255, update)  # shift_value 조정

# 초기 이미지 처리
process_image()

cv2.waitKey(0)
cv2.destroyAllWindows()
