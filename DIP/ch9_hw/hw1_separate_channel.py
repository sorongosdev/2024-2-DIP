import cv2

# 이미지 불러오기
image = cv2.imread('images/opencv.png')

# 이미지가 제대로 불러와졌는지 확인
if image is None:
    print("이미지를 불러오는 데 실패했습니다.")
else:
    # 이미지 윈도우에 띄우기
    cv2.imshow('Loaded Image', image)

    # 키 입력 대기 (무한 대기)
    cv2.waitKey(0)

    # 모든 윈도우 종료
    cv2.destroyAllWindows()
