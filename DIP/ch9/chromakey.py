import cv2

img = cv2.imread("images/chroma.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("images/chroma.jpg", cv2.IMREAD_COLOR)

converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

greenScreen = cv2.inRange(converted, (60 - 10, 100, 100), (60 + 10, 255, 255))

inverted = cv2.bitwise_not(greenScreen)

dst = cv2.bitwise_and(img, img, mask=inverted)
dst1 = cv2.bitwise_or(dst, img2, mask=greenScreen)
final_result = cv2.bitwise_or(dst, dst1)

cv2.imshow("Original Image", img)
cv2.imshow("Green Screen Mask", greenScreen)
cv2.imshow("Result without Green Screen", dst)
cv2.imshow("Final Result", final_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
