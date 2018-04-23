#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
import urllib
import os
import bz2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from ros_peoplemodel.srv import DlibShapes
from ros_peoplemodel.srv import DlibShapesResponse

DLIB_SHAPE_MODEL_FILE = expanduser("~/.dlib/shape_predictor.dat")
DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


def initializeModel():
    urlOpener = urllib.URLopener()
    if not os.path.exists(expanduser("~/.dlib")):
        os.makedirs(expanduser("~/.dlib"))

    if not os.path.isfile(DLIB_SHAPE_MODEL_FILE):
        print("downloading %s" % DLIB_SHAPE_MODEL_URL)
        urlOpener.retrieve(DLIB_SHAPE_MODEL_URL, DLIB_SHAPE_MODEL_FILE)
        data = bz2.BZ2File(DLIB_SHAPE_MODEL_FILE).read() # get the decompressed data
        open(DLIB_SHAPE_MODEL_FILE, 'wb').write(data) # write a uncompressed file

def handleRequest(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    d = dlib.rectangle(0, 0, image.shape[0], image.shape[1])

    shape = dlib_shape_predictor(image, d)
    shape_as_points = [Point(p.x, p.y, 0) for p in shape.parts()]

    return DlibShapesResponse(shape_as_points)

if __name__ == "__main__":
    initializeModel()
    bridge = CvBridge()
    dlib_shape_predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)

    rospy.init_node('vis_srv_dlib_shapes_server')
    srv = rospy.Service('vis_srv_dlib_shapes', DlibShapes, handleRequest)

    rospy.spin()
