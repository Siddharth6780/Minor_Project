import cv2
from cv2 import findContours
from cv2 import imshow
import numpy as np

def findAndPrintContours(blue_mask, blue):
    contours,_ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnts in contours:
        area = cv2.contourArea(cnts)
        (x, y, w, h) = cv2.boundingRect(cnts)
        cv2.putText(frame, "Status: {}".format('Color Detected'), (20, 25), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
        extractedContour = frame[y:y+h,x:x+w]
        cv2.imshow('cutted contour', extractedContour)
        cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    cv2.imshow("Blue Mask", blue)

while True:
    cap = cv2.VideoCapture('2.mp4');
    isclosed=0
    while (True):
        ret,frame = cap.read()

        if ret == True:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Blue Color
            low_blue = np.array([63, 72, 49])
            high_blue = np.array([153, 255, 255])
            blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
            blue = cv2.bitwise_and(frame, frame, mask = blue_mask)
            blue_mask = cv2.medianBlur(blue_mask, 5)

            findAndPrintContours(blue_mask, blue)

            key = cv2.waitKey(1)
            if key == 27:
                isclosed = 1
                break
        else:
            break
    if isclosed:
        break

cv2.destroyAllWindows()