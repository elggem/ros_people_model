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

DLIB_CNN_MODEL_FILE = "/tmp/dlib/mmod_cnn.dat"
DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"

CNN_SCALE = 0.4
CNN_FRATE = 1.0/5.0

def initializeModel():
    urlOpener = urllib.URLopener()
    if not os.path.exists("/tmp/dlib"):
        os.makedirs("/tmp/dlib")

    if not os.path.isfile(DLIB_CNN_MODEL_FILE):
        print("downloading %s" % DLIB_CNN_MODEL_URL)
        urlOpener.retrieve(DLIB_CNN_MODEL_URL, DLIB_CNN_MODEL_FILE)
        data = bz2.BZ2File(DLIB_CNN_MODEL_FILE).read() # get the decompressed data
        open(DLIB_CNN_MODEL_FILE, 'wb').write(data) # write a uncompressed file

def performCNNFaceDetection(img, scale=1.0):
    if scale is not 1.0:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # perform CNN detection
    cnnDets = dlib_cnn_detector(img, 1)
    # rescale
    return [dlib.rectangle(top    = int(d.rect.top()    / scale),
                    rospy.Timer(rospy.Duration(FRONTAL_FRATE), faceDetectFrontalCallback)
    rospy.Timer(rospy.Duration(ANALYSIS_FRATE), faceAnalysis)
           bottom = int(d.rect.bottom() / scale),
                           left   = int(d.rect.left()   / scale),
                           right  = int(d.rect.right()  / scale)) for d in cnnDets]

def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")

def faceDetectCNNCallback(event):
    global IMAGE, FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = performCNNFaceDetection(IMAGE, scale=CNN_SCALE)



if __name__ == "__main__":
    initializeModel()
    rospy.init_node('vis_dlib_cnn', anonymous=True)
    bridge = CvBridge()

    # Publishers
    pub = rospy.Publisher('vis_dlib_cnn', Feature, queue_size=10)
    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)

    # Dlib
    dlib_cnn_detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_MODEL_FILE)
    # Launch detectors
    rospy.Timer(rospy.Duration(CNN_FRATE), faceDetectCNNCallback)

    rospy.spin()
