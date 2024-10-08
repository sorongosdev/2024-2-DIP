import numpy as np, cv2


def draw_histo(hist, shape = (200, 255 * 2)):
    hist_img = np.full(shape, 255, np.uint8)
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX)
    gap = hist_img.shape[1] / hist.shape[0]

    for i, h in enumerate(hist):
        x = int(round(i * gap))
        w = int(round(gap))
        roi = (x, 0, w, int(h))
        cv2.rectangle(hist_img, roi, 0, cv2.FILLED)

    # 가로축 레이블 추가
    for i in range(hist.shape[0]):
        x = int(round(i * gap)) + int(gap // 2)  # 각 막대의 중앙 x 위치
        cv2.putText(hist_img, str(i * (256 // hist.shape[0])), (x, shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    return cv2.flip(hist_img, 0)

    # return cv2.flip(hist_img, 0)


image = cv2.imread('images/crayfish.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception('Could not load image')

bsize, ranges = [64], [0, 256]
hist = cv2.calcHist([image], [0], None, bsize, ranges)

print("Histogram values:", hist.flatten())  # 히스토그램의 값을 출력

hist_img = draw_histo(hist, (200, 360))

print("Histogram image shape:", hist_img.shape)  # 히스토그램 이미지의 크기를 출력

cv2.imshow("img", image)
cv2.imshow("hist_img", hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
