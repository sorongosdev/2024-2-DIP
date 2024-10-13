import numpy as np
import cv2


def draw_histo_combined(hists, shape=(200, 255 * 2), colors=((255, 0, 0), (0, 255, 0), (0, 0, 255)), alpha=0.5):
    hist_img = np.full((*shape, 3), 255, np.uint8)  # 흰색 배경 생성

    for i, hist in enumerate(hists):
        cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
        gap = hist_img.shape[1] / hist.shape[0]

        for j in range(len(hist)):
            h = int(hist[j])  # 높이 계산
            x = int(round(j * gap))
            w = int(round(gap))

            # 막대 그래프 색상으로 투명도 적용
            overlay = np.full((shape[0], w, 3), colors[i], dtype=np.uint8)  # 막대 색상 생성
            if h > 0:  # 높이가 0보다 클 때만 그리기
                # 투명도 적용
                cv2.addWeighted(hist_img[shape[0] - h:, x:x + w], 1 - alpha,
                                overlay[:h, :], alpha, 0,
                                hist_img[shape[0] - h:, x:x + w])

    return hist_img  # flip을 제거하여 뒤집힌 문제 해결


def draw_overlap_histogram(hists, shape=(200, 255 * 2), colors=((255, 0, 0), (0, 255, 0), (0, 0, 255))):
    overlap_hist = np.zeros(hists[0].shape, dtype=np.float32)  # 겹쳐진 히스토그램 초기화
    for hist in hists:
        overlap_hist += hist  # 각 색상 채널의 히스토그램 합산

    # 겹쳐진 히스토그램 이미지 생성
    overlap_img = np.full(shape, 255, dtype=np.uint8)  # 흰색 배경
    cv2.normalize(overlap_hist, overlap_hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = overlap_img.shape[1] / overlap_hist.shape[0]

    for j in range(len(overlap_hist)):
        h = int(overlap_hist[j])  # 높이 계산
        x = int(round(j * gap))
        w = int(round(gap))

        if h > 0:  # 높이가 0보다 클 때만 그리기
            cv2.rectangle(overlap_img, (x, shape[0]), (x + w, shape[0] - h), (0, 0, 0), cv2.FILLED)

    return overlap_img


image = cv2.imread('images/crayfish.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('Could not load image')

bsize, ranges = [64], [0, 256]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR 순서로 정의

# 각 색상 채널에 대해 히스토그램 계산
hists = []
for i in range(len(colors)):
    hist = cv2.calcHist([image], [i], None, bsize, ranges)
    hists.append(hist.flatten())  # 1차원 배열로 변환하여 추가

# 겹쳐진 히스토그램 이미지 생성
hist_img_combined = draw_histo_combined(hists, (200, 360), colors, alpha=0.5)

# 겹쳐진 부분의 히스토그램 이미지 생성
overlap_hist_img = draw_overlap_histogram(hists, (200, 360), colors)

# 결과 출력
cv2.imshow("img", image)
cv2.imshow("Combined Histogram with Transparency", hist_img_combined)
cv2.imshow("Overlap Histogram", overlap_hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
