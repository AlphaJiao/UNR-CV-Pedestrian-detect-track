
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils 
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--videos", help = "path to the video file")
#args = vars(ap.parse_args())

#winStride = eval(args["win_stride"])
#padding = eval(args["padding"])
#meanShift = True if args["mean_shift"] > 0 else False

#if not args.get("videos", False):
    #camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0)
#else:
    #camera = cv2.VideoCapture(args["videos"])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

while(True):
    ret, frame = camera.read()

    if not ret: 
        continue
    frame = imutils.resize(frame, width=min(400, frame.shape[1])) 
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),padding=(32,32),scale=1.2)
    rects = np.array([[x, y, x + w, y + h]for (x, y, w, h) in rects])
    
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0),2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
