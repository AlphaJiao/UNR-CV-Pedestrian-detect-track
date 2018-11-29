#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils 
import argparse

#pub = rospy.Publisher('/Human/box', Image, queue_size=1) 
    
     
def callback(data):
    pub = rospy.Publisher('/Human/box', Image, queue_size=1)
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(data,"bgr8")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
    frame = imutils.resize(frame, width=min(400, frame.shape[1])) 
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),padding=(32,32),scale=1.2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0),2)
    cv2.imshow("talker", frame)
    cv2.waitKey(3)
    pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    #rate.sleep()

def listener():
    rospy.init_node('listener', anonymous = True)
    rospy.Subscriber('/cam0/image_raw',Image,callback)
#/usb_cam/image_raw
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass


