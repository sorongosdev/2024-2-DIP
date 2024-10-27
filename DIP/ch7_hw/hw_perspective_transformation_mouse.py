import numpy as np
import cv2

# 클릭한 좌표를 저장할 리스트
clicked_points = []


# 마우스 클릭 이벤트를 처리할 콜백 함수
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭
        if len(clicked_points) < 4:  # 4번 클릭까지 허용
            clicked_points.append((x, y))
            print(f"Clicked coordinates: ({x}, {y})")
            # 클릭한 위치에 원 그리기
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Original image", image)

            # 4번 클릭 후 좌표 출력
            if len(clicked_points) == 4:
                print("Four points clicked:", clicked_points)
                # np.float32 배열로 변환
                points_array = np.array(clicked_points, dtype=np.float32)
                print("Points as np.float32 array:", points_array)


# 이미지 로드
image = cv2.imread("images/perspective.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise Exception("Could not load image")

# 윈도우 생성 및 이미지 표시
cv2.imshow("Original image", image)

# 마우스 콜백 함수 설정
cv2.setMouseCallback("Original image", click_event)

# 키 입력 대기
cv2.waitKey(0)
cv2.destroyAllWindows()
