#!/usr/bin/python
import bz2
import os
import urllib
from os.path import expanduser

import cv2
import dlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from ros_peoplemodel.msg import Feature
from ros_peoplemodel.msg import Features
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

IMAGE = None

DLIB_CNN_MODEL_FILE = expanduser("~/.dlib/mmod_cnn.dat")
DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"


def initialize_model():
    urlOpener = urllib.URLopener()
    if not os.path.exists(expanduser("~/.dlib")):
        os.makedirs(expanduser("~/.dlib"))

    if not os.path.isfile(DLIB_CNN_MODEL_FILE):
        print("downloading %s" % DLIB_CNN_MODEL_URL)
        urlOpener.retrieve(DLIB_CNN_MODEL_URL, DLIB_CNN_MODEL_FILE)
        data = bz2.BZ2File(DLIB_CNN_MODEL_FILE).read()  # get the decompressed data
        open(DLIB_CNN_MODEL_FILE, 'wb').write(data)  # write a uncompressed file


def perform_cnn_face_detection(img, scale=1.0):
    if scale is not 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # perform CNN detection
    cnn_dets = dlib_cnn_detector(img, 1)
    # rescale
    return [dlib.rectangle(top=int(d.rect.top() / scale),
                           bottom=int(d.rect.bottom() / scale),
                           left=int(d.rect.left() / scale),
                           right=int(d.rect.right() / scale)) for d in cnn_dets]


def image_callback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")


def face_detect_cnn_callback(event):
    global IMAGE

    if IMAGE is None:
        return

    cnn_results = perform_cnn_face_detection(IMAGE, scale=CNN_SCALE)

    features = Features()
    try:
        features.image = bridge.cv2_to_imgmsg(np.array(IMAGE))
    except Exception:
        pass
    features.features = []

    for k, d in enumerate(cnn_results):
        padding = int(IMAGE.shape[0] * CNN_PADDING)

        feature = Feature()

        roi = RegionOfInterest()
        roi.x_offset = np.maximum(d.left() - padding, 0)
        roi.y_offset = np.maximum(d.top() - padding, 0)
        roi.height = np.minimum(d.bottom() - roi.y_offset + padding, IMAGE.shape[0])
        roi.width = np.minimum(d.right() - roi.x_offset + padding, IMAGE.shape[1])

        feature.roi = roi
        feature.crop = bridge.cv2_to_imgmsg(np.array(IMAGE[roi.y_offset:roi.y_offset + roi.height,
                                                     roi.x_offset:roi.x_offset + roi.width, :]))

        features.features.append(feature)

    pub.publish(features)


if __name__ == "__main__":
    initialize_model()
    rospy.init_node('vis_dlib_cnn', anonymous=True)
    bridge = CvBridge()

    CNN_SCALE = rospy.get_param('~scale', 0.4)
    CNN_FRATE = 1.0 / rospy.get_param('~fps', 5.0)
    CNN_PADDING = rospy.get_param('~padding', 0.1)

    # Publishers
    pub = rospy.Publisher('/vis_dlib_cnn', Features, queue_size=10)
    # Subscribers
    rospy.Subscriber(rospy.get_param('~topic_name', '/camera/image_raw'), Image, image_callback)

    # Dlib
    dlib_cnn_detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_MODEL_FILE)
    # Launch detectors
    rospy.Timer(rospy.Duration(CNN_FRATE), face_detect_cnn_callback)

    rospy.spin()
