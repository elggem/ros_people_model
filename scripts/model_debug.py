#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ros_peoplemodel.msg import Features

DRAW_FRAMERATE = 1.0/30.0

IMAGE = None
FACES = None

EMOTIONS = {
    0 : "anger",
    1 : "disgust",
    2 : "fear",
    3 : "happy",
    4 : "sad",
    5 : "surprise",
    6 : "neutral"
}

def debugDraw(self):
    global IMAGE, FACES

    if IMAGE is None:
        return

    cnn_clr = (0, 0, 255)
    frt_clr = (0, 0, 0)
    txt_clr = (255, 255, 255)
    shp_clr = (255, 255, 255)
    emo_clr = (150, 150, 125)

    frame = IMAGE.copy()
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)


    cv2.imshow("Image",frame)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        return


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")

def facesCallback(data):
    global FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = data

if __name__ == "__main__":
    rospy.init_node('model_debug_output', anonymous=True)
    bridge = CvBridge()

    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)
    rospy.Subscriber("/faces", Features, cnnCallback)

    # Launch drawing timer
    rospy.Timer(rospy.Duration(DRAW_FRAMERATE), debugDraw)

    rospy.spin()
