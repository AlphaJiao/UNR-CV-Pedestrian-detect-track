#!/usr/bin/env python 
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils 
import argparse

def callback(data):
    br = CvBridge()
    frame = br.imgmsg_to_cv2(data,"bgr8")
    rospy.loginfo('receiving image')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    frame = imutils.resize(frame, width=min(400, frame.shape[1])) 
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),padding=(32,32),scale=1.2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0),2)
    
    cv2.imshow("camera",frame)
    cv2.waitKey(1)


def listener():
    rospy.init_node('listener', anonymous = True)
    rospy.Subscriber('/cam0/image_raw',Image,callback)
    #rospy.Subscriber('/cam_usb/image_raw',Image,callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener()
