#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def callback(imgmsg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imgmsg, "bgr8")
    cv2.imshow("listener", img)
    cv2.waitKey(3)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/Human/box", Image, callback)
    rospy.spin()
if __name__ == '__main__':
    listener()
