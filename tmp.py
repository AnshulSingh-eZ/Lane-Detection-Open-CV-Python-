import cv2

cap = cv2.VideoCapture(0)
while(True):
    succ, frame = cap.read()
    cannyimg = cv2.Canny(frame, 100,50)
    cv2.imshow("output", cannyimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release();
cv2.destroyAllWindows();