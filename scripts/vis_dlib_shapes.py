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

DLIB_SHAPE_MODEL_FILE = "/tmp/dlib/shape_predictor.dat"
DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

ANALYSIS_FRATE = 1.0/30.0

def initializeModel():
    urlOpener = urllib.URLopener()
    if not os.path.exists("/tmp/dlib"):
        os.makedirs("/tmp/dlib")

    if not os.path.isfile(DLIB_SHAPE_MODEL_FILE):
        print("downloading %s" % DLIB_SHAPE_MODEL_URL)
        urlOpener.retrieve(DLIB_SHAPE_MODEL_URL, DLIB_SHAPE_MODEL_FILE)
        data = bz2.BZ2File(DLIB_SHAPE_MODEL_FILE).read() # get the decompressed data
        open(DLIB_SHAPE_MODEL_FILE, 'wb').write(data) # write a uncompressed file


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")


def faceAnalysis(event):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL, FACE_CANDIDATES_SIDEWAYS

    for k, d in enumerate(FACE_CANDIDATES_FRONTAL):
        face = Face()
        cropped_face = IMAGE[d.top():d.bottom(), d.left():d.right(), :]

        if len(cropped_face)==0 or len(cropped_face[0])==0:
            continue

        face.image = bridge.cv2_to_imgmsg(np.array(cropped_face))
        face.bounding_box = [d.top(), d.bottom(), d.left(), d.right()]

        # Get the shape
        shape = dlib_shape_predictor(IMAGE, d)
        face.shape = [Point(p.x, p.y, 0) for p in shape.parts()]

        ## IN CASE WE WANNA SEE IT
        faces.append(face)
        pub.publish(face)



if __name__ == "__main__":
    initializeModel()
    initializeFaceID()

    rospy.init_node('vis_dlib_shapes', anonymous=True)

    bridge = CvBridge()
    # Publishers
    pub = rospy.Publisher('vis_dlib_shapes', Shape, queue_size=10)
    # Subscribers
    rospy.Subscriber("/vis_dlib_frontal", Feature, imageCallback)

    # Dlib
    dlib_shape_predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)

    # Launch detectors
    rospy.Timer(rospy.Duration(ANALYSIS_FRATE), faceAnalysis)

    rospy.spin()
