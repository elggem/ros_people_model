#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from ros_peoplemodel.msg import Features
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import Image

FACE_CANDIDATES_CNN = None

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


def featuresCallback(features):
    global FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = features


def faceDetectFrontalCallback(event):
    global FACE_CANDIDATES_CNN

    if FACE_CANDIDATES_CNN is None:
        return

    IMAGE = bridge.imgmsg_to_cv2(FACE_CANDIDATES_CNN.image, "8UC3")

    features = Features()
    features.image = FACE_CANDIDATES_CNN.image
    features.crops = []
    features.rois = []

    #goes through list and only saves the one
    for k, crop_img in enumerate(FACE_CANDIDATES_CNN.crops):
        crop = bridge.imgmsg_to_cv2(crop_img, "8UC3")
        dets = performFaceDetection(crop, scale=FRONTAL_SCALE)

        if len(dets)==1:
            d = dets[0]

            roi = RegionOfInterest()
            roi.x_offset = np.maximum(FACE_CANDIDATES_CNN.rois[k].x_offset + d.left(), 0)
            roi.y_offset = np.maximum(FACE_CANDIDATES_CNN.rois[k].y_offset + d.top(), 0)
            roi.height =   np.maximum(d.bottom() - d.top(), 0)
            roi.width =    np.maximum(d.right() - d.left(), 0)

            features.rois.append(roi)
            features.crops.append(bridge.cv2_to_imgmsg(np.array(IMAGE[roi.y_offset:roi.y_offset+roi.height,
                                                                     roi.x_offset:roi.x_offset+roi.width, :])))

    pub.publish(features)

if __name__ == "__main__":
    rospy.init_node('vis_dlib_frontal', anonymous=True)
    bridge = CvBridge()

    FRONTAL_SCALE = rospy.get_param('~scale', 0.4)
    FRONTAL_FRATE = rospy.get_param('~rate', 1.0/8.0)

    # Publishers
    pub = rospy.Publisher('vis_dlib_frontal', Features, queue_size=10)
    # Subscribers
    rospy.Subscriber(rospy.get_param('~topic_name', '/people/vis_dlib_cnn'), Features, featuresCallback)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()

    # Launch detectors
    rospy.Timer(rospy.Duration(FRONTAL_FRATE), faceDetectFrontalCallback)

    rospy.spin()
