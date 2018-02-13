#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ros_slopp.msg import Face

FRONTAL_SCALE = 0.4
FRONTAL_FRATE = 1.0/8.0


def performFaceDetection(img, scale=1.0):
    if scale is not 1.0:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # perform CNN detection
    dets = dlib_detector(img, 1)
    # rescale
    return [dlib.rectangle(top    = int(d.top()    / scale),
                           bottom = int(d.bottom() / scale),
                           left   = int(d.left()   / scale),
                           right  = int(d.right()  / scale)) for d in dets]


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")


def faceDetectFrontalCallback(event):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL, FACE_CANDIDATES_SIDEWAYS

    sideways_dets = []
    frontal_dets = []
    #goes through list and only saves the one
    for k, d in enumerate(FACE_CANDIDATES_CNN):
        padding = int(IMAGE.shape[0]*0.1)
        t = np.maximum(d.top()  - padding, 0)
        l = np.maximum(d.left() - padding, 0)
        b = np.minimum(d.bottom() + padding, IMAGE.shape[0])
        r = np.minimum(d.right()  + padding, IMAGE.shape[1])
        cropped_face = IMAGE[t:b, l:r, :]

        dets = performFaceDetection(cropped_face, scale=FRONTAL_SCALE)

        if len(dets)==1:
            frontal_dets.append(dlib.rectangle(   top = t + dets[0].top(),
                                               bottom = t + dets[0].bottom(),
                                                 left = l + dets[0].left(),
                                                right = l + dets[0].right()))
        else:
            sideways_dets.append(d)

    FACE_CANDIDATES_FRONTAL = frontal_dets
    FACE_CANDIDATES_SIDEWAYS = sideways_dets



if __name__ == "__main__":
    initializeModel()

    rospy.init_node('vis_dlib_frontal', anonymous=True)

    bridge = CvBridge()

    # Publishers
    pub = rospy.Publisher('vis_dlib_frontal', Face, queue_size=10)
    # Subscribers
    rospy.Subscriber("/vis_dlib_cnn", Feature, featureCallback)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()

    # Launch detectors
    rospy.Timer(rospy.Duration(FRONTAL_FRATE), faceDetectFrontalCallback)

    rospy.spin()
