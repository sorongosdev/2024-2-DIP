import numpy as np
import cv2

low_th = 100
high_th = low_th * 2


def nonmax_suppression(sobel, direct):
    rows, cols = sobel.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            values = sobel[i - 1:i + 2, j - 1:j + 2].flatten()
            first = [3, 0, 1, 2]
            id = first[direct[i, j]]
            v1, v2 = values[id], values[8 - id]
            dst[i, j] = sobel[i, j] if (v1 < sobel[i, j] > v2) else 0
    return dst


def trace(max_sobel, i, j, low, pos_ck, canny):
    h, w = max_sobel.shape
    if (0 <= i < h and 0 <= j < w) == False: return
    if pos_ck[i, j] == 0 and max_sobel[i, j] > low:
        pos_ck[i, j] = 255
        canny[i, j] = 255
        trace(max_sobel, i - 1, j - 1, low, pos_ck, canny)
        trace(max_sobel, i, j - 1, low, pos_ck, canny)
        trace(max_sobel, i + 1, j - 1, low, pos_ck, canny)
        trace(max_sobel, i - 1, j, low, pos_ck, canny)
        trace(max_sobel, i + 1, j, low, pos_ck, canny)
        trace(max_sobel, i - 1, j + 1, low, pos_ck, canny)
        trace(max_sobel, i, j + 1, low, pos_ck, canny)
        trace(max_sobel, i + 1, j + 1, low, pos_ck, canny)


def hysteresis_th(max_sobel, low, high, pos_ck, canny):
    rows, cols = max_sobel.shape[:2]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if max_sobel[i, j] > high:
                trace(max_sobel, i, j, low, pos_ck, canny)


# 이미지 읽기
image = cv2.imread("images/color_edge.jpg")
if image is None: raise Exception("No image")

# 결과 저장을 위한 배열
channels_canny = []

# 각 채널에 대해 캐니 엣지 적용
channel_names = ['blue', 'green', 'red']
for i, channel_name in enumerate(channel_names):  # B, G, R 채널
    channel = image[:, :, i]

    # 가우시안 블러
    gaus_img = cv2.GaussianBlur(channel, (5, 5), 0.3)
    Gx = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 1, 0, 3)
    Gy = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 0, 1, 3)

    sobel = np.fabs(Gx) + np.fabs(Gy)
    directs = cv2.phase(Gx, Gy) / (np.pi / 4)
    directs = directs.astype(int) % 4
    max_sobel = nonmax_suppression(sobel, directs)

    pos_ck = np.zeros(channel.shape, np.uint8)
    canny = np.zeros(channel.shape, np.uint8)

    hysteresis_th(max_sobel, low_th, high_th, pos_ck, canny)

    channels_canny.append(canny)

# 채널별 결과 출력
cv2.imshow("Original Image", image)
for idx, (canny_result, channel_name) in enumerate(zip(channels_canny, channel_names)):
    cv2.imshow(f"Canny Edge - {channel_name.capitalize()}", canny_result)

# 세 개의 채널 결과 합치기
merged_canny = cv2.merge(channels_canny)
cv2.imshow("Merged Canny Edges", merged_canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
