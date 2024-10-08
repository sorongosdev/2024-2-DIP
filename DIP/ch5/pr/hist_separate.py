import cv2


def onThreshold(value):
    result = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("result", result)


image = cv2.imread("images/color_space.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Could not load image")

cv2.namedWindow("result")
cv2.createTrackbar("threshold", "result", 128, 255, onThreshold)
onThreshold(128)
cv2.imshow("image", image)
cv2.waitKey()
