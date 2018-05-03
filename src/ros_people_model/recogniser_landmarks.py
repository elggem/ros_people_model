#!/usr/bin/python
import bz2
import os
import urllib
from os.path import expanduser

import dlib
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np


class LandmarkRecogniser(object):
    DLIB_SHAPE_MODEL_FILE = expanduser("~/.dlib/shape_predictor.dat")
    DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    def __init__(self):
        self.bridge = CvBridge()
        self.dlib_shape_predictor = dlib.shape_predictor(LandmarkRecogniser.DLIB_SHAPE_MODEL_FILE)

    def initialize_models(self):
        url_opener = urllib.URLopener()
        if not os.path.exists(expanduser("~/.dlib")):
            os.makedirs(expanduser("~/.dlib"))

        if not os.path.isfile(LandmarkRecogniser.DLIB_SHAPE_MODEL_FILE):
            rospy.loginfo("downloading %s" % LandmarkRecogniser.DLIB_SHAPE_MODEL_URL)
            url_opener.retrieve(LandmarkRecogniser.DLIB_SHAPE_MODEL_URL, LandmarkRecogniser.DLIB_SHAPE_MODEL_FILE)
            data = bz2.BZ2File(LandmarkRecogniser.DLIB_SHAPE_MODEL_FILE).read()  # get the decompressed data
            open(LandmarkRecogniser.DLIB_SHAPE_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

    def recognize(self, image):
        d = dlib.rectangle(0, 0, image.shape[0], image.shape[1])

        return self.dlib_shape_predictor(image, d)

    @staticmethod
    def shapes_to_geometry_msgs_points(landmarks):
        return [Point(p.x, p.y, 0) for p in landmarks.parts()]