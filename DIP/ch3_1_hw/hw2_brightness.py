import cv2

# 이미지 읽기
image = cv2.imread('images/gray_img.jpg')

while True:
    key = cv2.waitKeyEx(100)
    if key == 27:
        break  # ESC

    # 'r' 키가 눌리면 이미지 저장
    elif key == ord('r'):
        cv2.imwrite("images/brightness_img.jpg", image)

    # 왼쪽 화살표
    elif key == 2424832:
        # 블루 채널 밝기 감소
        image[:, :, 0] -= 20
        
    # 오른쪽 화살표
    elif key == 2555904:
        # 블루 채널 밝기 증가
        image[:, :, 0] += 20

    # 이미지를 업데이트하여 표시
    cv2.imshow("HW2", image)

cv2.destroyAllWindows()
