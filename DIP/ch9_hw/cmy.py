import cv2
import numpy as np

img = cv2.imread("images/opencv.png", cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cmy = 1 - img_rgb / 255.0

cyan_mask = cv2.inRange(cmy[:, :, 0], 0.0, 0.2)

magenta_mask = cv2.inRange(cmy[:, :, 1], 0.0, 0.2)

yellow_mask = cv2.inRange(cmy[:, :, 2], 0.0, 0.2)

cyan_mask_inverted = cv2.bitwise_not(cyan_mask)
magenta_mask_inverted = cv2.bitwise_not(magenta_mask)
yellow_mask_inverted = cv2.bitwise_not(yellow_mask)

cv2.imshow("Cyan channel", cyan_mask_inverted)
cv2.imshow("Magenta channel", magenta_mask_inverted)
cv2.imshow("Yellow channel", yellow_mask_inverted)

cv2.waitKey(0)
cv2.destroyAllWindows()
