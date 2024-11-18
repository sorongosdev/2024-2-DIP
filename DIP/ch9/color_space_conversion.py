import cv2

img = cv2.imread("images/opencv.png")

img_color = cv2.applyColorMap(img, cv2.COLORMAP_HOT)

cv2.imshow("img_color", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
