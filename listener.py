#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import imutils 
from cv_bridge import CvBridge

def callback(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    frame = imutils.resize(img, width=min(480, img.shape[1]))
    cv2.imshow("listener", frame)
    cv2.waitKey(3)

def listener():
    rospy.init_node('listener', anonymous=True)
    #rospy.Subscriber('tracker/raw_im',Image, callback)
    #rospy.Subscriber('tracker/thresh_sub',Image, callback)
    rospy.Subscriber('tracker/ptcls', Image, callback)



    rospy.spin()
if __name__ == '__main__':
    listener()
