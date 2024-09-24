import cv2

image = cv2.imread("../images/dog.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상파일 읽기 오류")
if image.ndim != 3: raise Exception("컬러 영상 아님")

bgr = cv2.split(image)
print("bgr 자료형:", type(bgr), type(bgr[0]), type(bgr[0][0][0]))
