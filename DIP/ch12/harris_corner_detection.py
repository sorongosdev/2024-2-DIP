import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image
image = cv2.imread("images/checker.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# harris corner detection
gray = np.float32(gray)
harris_response = cv2.cornerHarris(gray, 2, 3, 0.04)

harris_response = cv2.dilate(harris_response, None)

# thresholding
threshold = 0.01 * harris_response.max()
image[harris_response > threshold] = [0, 0, 255]  # red corner

plt.figure(figsize = (10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")
plt.axis('off')
plt.show()

# complete
