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


def calculate_perspective_transform(points_array):
    """ 변환 행렬 계산 """
    w, h = 470, 400
    pts2 = np.float32(
        [(w * 0.25, h * 0.25), (w * 0.75, h * 0.25),
         (w * 0.25, h * 0.75), (w * 0.75, h * 0.75)]
    )
    perspect_mat = cv2.getPerspectiveTransform(points_array, pts2)
    return perspect_mat, pts2


def warp_image(image, perspect_mat):
    """ 행렬을 받아서 이미지를 왜곡하는 함수 """
    dst = cv2.warpPerspective(image, perspect_mat, image.shape[1::-1], cv2.INTER_CUBIC)
    return dst


def process_perspective_transform():
    """투사 변환하는 함수"""

    global pts1
    points_array = np.array(pts1, dtype=np.float32)
    print("Points as np.float32 array:", points_array)

    # 변환 행렬 계산
    perspect_mat, pts2 = calculate_perspective_transform(points_array)
    dst = warp_image(image, perspect_mat)
    print("[perspect_mat] = \n%s\n" % perspect_mat)

    # 결과 이미지 표시
    cv2.imshow("Warped Image", dst)


""" 마우스 콜백 함수"""


def click_event(event, x, y, flags, param):
    global pts1  # pts1을 전역 변수로 사용
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        if len(pts1) < 4:  # 4번 클릭까지 허용
            pts1.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")
            # 클릭한 위치에 원 그리기
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Original image", image)

            # 4번 클릭 후 좌표 출력, 변환
            if len(pts1) == 4:
                print("Four points clicked:", pts1)
                process_perspective_transform()


# 윈도우 생성 및 이미지 표시
cv2.imshow("Original image", image)

# 마우스 콜백 설정
cv2.setMouseCallback("Original image", click_event)

# 키 입력 대기
cv2.waitKey(0)
cv2.destroyAllWindows()
