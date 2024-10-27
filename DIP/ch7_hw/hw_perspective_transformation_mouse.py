import numpy as np
import cv2

# 클릭한 좌표를 저장할 리스트
pts1 = []

# 이미지 로드
image = cv2.imread("images/perspective2.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise Exception("Could not load image")
else:
    print("Loaded image")


# 마우스 클릭 이벤트를 처리할 콜백 함수
def click_event(event, x, y, flags, param):
    global pts1  # pts1을 전역 변수로 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        if len(pts1) < 4:  # 4번 클릭까지 허용
            pts1.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")
            # 클릭한 위치에 원 그리기
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Original image", image)

            # 4번 클릭 후 좌표 출력 및 변환 수행
            if len(pts1) == 4:
                print("Four points clicked:", pts1)
                # np.float32 배열로 변환
                points_array = np.array(pts1, dtype=np.float32)
                print("Points as np.float32 array:", points_array)

                # 변환 행렬 계산 및 적용
                w, h = 470, 400
                pts2 = np.float32(
                    [(w * 0.25, h * 0.25), (w * 0.75, h * 0.25), (w * 0.25, h * 0.75), (w * 0.75, h * 0.75)])
                perspect_mat = cv2.getPerspectiveTransform(points_array, pts2)
                dst = cv2.warpPerspective(image, perspect_mat, image.shape[1::-1], cv2.INTER_CUBIC)
                print("[perspect_mat] = \n%s\n" % perspect_mat)

                # 동차 좌표 처리
                ones = np.ones((4, 1), np.float64)
                pts3 = np.append(points_array, ones, axis=1)
                pts4 = cv2.gemm(pts3, perspect_mat.T, 1, None, 1)

                print(" 원본 영상 좌표 \t 목적 영상 좌표 \t\t 동차 좌표 \t\t 변환 결과 좌표")

                for i in range(len(pts4)):
                    pts4[i] /= pts4[i][2]
                    print("%i : %-14s %-14s %-18s%-18s" % (i, points_array[i], pts2[i], pts3[i], pts4[i]))
                    cv2.circle(image, tuple(points_array[i].astype(int)), 4, (0, 255, 0), -1)
                    cv2.circle(dst, tuple(pts2[i].astype(int)), 4, (0, 255, 0), -1)

                # 결과 이미지 표시
                cv2.imshow("Warped Image", dst)


# 윈도우 생성 및 이미지 표시
cv2.imshow("Original image", image)

# 마우스 콜백 함수 설정
cv2.setMouseCallback("Original image", click_event)

# 키 입력 대기
cv2.waitKey(0)
cv2.destroyAllWindows()
