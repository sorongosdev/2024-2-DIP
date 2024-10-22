import time

import numpy as np, cv2
from scaling import scaling, scaling2


def time_check(func, image, size, title):
    start_time = time.perf_counter()
    ret_img = func(image, size)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(title, " 수행시간 = %%0.2f ms" % elapsed)
    return ret_img


image = cv2.imread("images/scaling.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("No image")

dst1 = scaling(image, (150, 200))
dst2 = scaling2(image, (150, 200))
dst3 = time_check(scaling, image, (300, 400), "[방법1] 좌표행렬 방식 >")
dst4 = time_check(scaling2, image, (300, 400), "[방법2] 반복문 방식 >")

cv2.imshow("image", image)
cv2.imshow("dst1 - zoom out", dst1)
cv2.imshow("dst3 - zoom out", dst3)
cv2.resizeWindow("dst1 - zoom out", 260, 200)
cv2.waitKey(0)
