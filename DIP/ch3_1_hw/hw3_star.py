import numpy as np
import cv2

a = 50


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

        triangle1 = np.array(
            [[x, y],
             [x - a / 2, y + 3 * a / 4],
             [x + a / 2, y + 3 * a / 4]],
            np.int32)

        triangle2 = np.array(
            [[x - a / 2, y + a / 4],
             [x + a / 2, y + a / 4],
             [x, y + a]],
            np.int32)

        cv2.polylines(image, [triangle1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(image, [triangle2], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("HW3", image)


    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.imwrite("images/star_dog.jpg", image)


# 강아지 사진
image = cv2.imread('images/dog.jpg')

cv2.imshow("HW3", image)

cv2.setMouseCallback("HW3", onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
