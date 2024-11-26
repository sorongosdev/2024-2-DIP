import cv2

threshold_value = 128
threshold_type = 0
max_value = 255
max_type = 4
max_binary_value = 255


def my_threshold(val):
    _, dst = cv2.threshold(src, threshold_value, max_binary_value, threshold_type)
    cv2.imshow("result", dst)


def update_threshold_value(val):
    global threshold_value
    threshold_value = val
    my_threshold(val)


def update_threshold_type(val):
    global threshold_type
    threshold_type = val
    my_threshold(val)


src = cv2.imread("images/lenna.jpg", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("No image")

cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar("Threshold Value", "result", threshold_value, max_value, update_threshold_value)
cv2.createTrackbar("Threshold Type", "result", threshold_value, max_value, update_threshold_type)

my_threshold(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
