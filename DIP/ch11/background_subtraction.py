import cv2

cap = cv2.VideoCapture("images/tennis_ball.mp4")

if not cap.isOpened():
    print("Could not open video file")
    exit()

fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 16, detectShadows = True)

fgbg_knn = cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold = 400, detectShadows = True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask_mog2 = fgbg_mog2.apply(frame)
    fgmask_knn = fgbg_knn.apply(frame)

    cv2.imshow("Oringinal Frame", frame)
    cv2.imshow("Foreground Mask - MOG2", fgmask_mog2)
    cv2.imshow("Foreground Mask - KNN", fgmask_knn)

    if cv2.waitKey(30) == 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
