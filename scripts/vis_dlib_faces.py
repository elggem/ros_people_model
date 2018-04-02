#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from ros_peoplemodel.msg import Feature
from ros_peoplemodel.msg import Features
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import Image

from ros_peoplemodel.srv import DlibFaceID
from ros_peoplemodel.srv import DlibShapes
from ros_peoplemodel.srv import iCogEmopy
from ros_peoplemodel.srv import iCogEyeState

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
    features.features = []

    #goes through list and only saves the one
    for k, feature in enumerate(FACE_CANDIDATES_CNN.features):
        crop = bridge.imgmsg_to_cv2(feature.crop, "8UC3")
        dets = performFaceDetection(crop, scale=FRONTAL_SCALE)

        if len(dets)==1:
            d = dets[0]

            ftr = Feature()

            roi = RegionOfInterest()
            roi.x_offset = max(feature.roi.x_offset + d.left(), 0)
            roi.y_offset = max(feature.roi.y_offset + d.top(), 0)
            roi.height =   max(d.bottom() - d.top(), 0)
            roi.width =    max(d.right() - d.left(), 0)

            ftr.roi = roi
            ftr.crop = bridge.cv2_to_imgmsg(np.array(IMAGE[roi.y_offset:roi.y_offset+roi.height,
                                                               roi.x_offset:roi.x_offset+roi.width, :]))

            try:
                ftr.shapes = srv_dlib_shapes(ftr.crop).shape
            except Exception:
                pass

            try:
                ftr.face_id = srv_dlib_faceid(ftr.crop, ftr.roi, ftr.shapes).face_id
            except Exception:
                pass

            try:
                ftr.emotions = srv_icog_emopy(ftr.crop).emotions
            except Exception:
                pass

            try:
                ftr.eyes_closed = srv_icog_eyestate(ftr.crop).eyes_closed
            except Exception:
                pass

            features.features.append(ftr)

    pub.publish(features)

if __name__ == "__main__":
    rospy.init_node('vis_dlib_frontal', anonymous=True)
    bridge = CvBridge()

    FRONTAL_SCALE = rospy.get_param('~scale', 0.4)
    FRONTAL_FRATE = 1.0/rospy.get_param('~fps', 5.0)

    # Publishers
    pub = rospy.Publisher('vis_dlib_frontal', Features, queue_size=10)
    # Subscribers
    rospy.Subscriber(rospy.get_param('~topic_name', '/vis_dlib_cnn'), Features, featuresCallback)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()

    # Attribute Services
    srv_dlib_shapes = rospy.ServiceProxy('vis_srv_dlib_shapes', DlibShapes, persistent=True)
    srv_dlib_faceid = rospy.ServiceProxy('vis_srv_dlib_id', DlibFaceID, persistent=False)
    srv_icog_emopy = rospy.ServiceProxy('vis_srv_icog_emopy', iCogEmopy, persistent=True)
    srv_icog_eyestate = rospy.ServiceProxy('vis_srv_icog_eyestate', iCogEyeState, persistent=True)

    #rospy.wait_for_service('vis_srv_dlib_shapes')
    #rospy.wait_for_service('vis_srv_dlib_id')
    #rospy.wait_for_service('vis_srv_icog_emopy')
    #rospy.wait_for_service('vis_srv_icog_eyestate')

    # Launch detectors
    rospy.Timer(rospy.Duration(FRONTAL_FRATE), faceDetectFrontalCallback)

    rospy.spin()
