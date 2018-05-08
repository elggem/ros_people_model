import bz2
import os
import urllib
from os.path import expanduser

import cv2
import dlib
import rospy


class FaceRecogniser(object):
    DLIB_CNN_MODEL_FILE = expanduser("~/.dlib/mmod_cnn.dat")
    DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"

    def __init__(self):

        # Dlib detectors
        self.dlib_face_detector = dlib.cnn_face_detection_model_v1(FaceRecogniser.DLIB_CNN_MODEL_FILE)
        self.dlib_frontal_face_detector = dlib.get_frontal_face_detector()

    def initialize_model(self):
        url_opener = urllib.URLopener()
        if not os.path.exists(expanduser("~/.dlib")):
            os.makedirs(expanduser("~/.dlib"))

        if not os.path.isfile(FaceRecogniser.DLIB_CNN_MODEL_FILE):
            rospy.loginfo("downloading %s" % FaceRecogniser.DLIB_CNN_MODEL_URL)
            url_opener.retrieve(FaceRecogniser.DLIB_CNN_MODEL_URL, FaceRecogniser.DLIB_CNN_MODEL_FILE)
            data = bz2.BZ2File(FaceRecogniser.DLIB_CNN_MODEL_FILE).read()  # get the decompressed data
            open(FaceRecogniser.DLIB_CNN_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

    def detect_faces(self, image, scale=1.0):
        if scale is not 1.0:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # perform CNN detection
        cnn_dets = self.dlib_face_detector(image, 1)

        # rescale
        return [dlib.rectangle(top=int(d.rect.top() / scale),
                               bottom=int(d.rect.bottom() / scale),
                               left=int(d.rect.left() / scale),
                               right=int(d.rect.right() / scale)) for d in cnn_dets]

    def detect_frontal_faces(self, image, scale=1.0):
        if scale is not 1.0:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # perform CNN detection
        dets = self.dlib_frontal_face_detector(image, 1)

        # rescale
        return [dlib.rectangle(top=int(d.top() / scale),
                               bottom=int(d.bottom() / scale),
                               left=int(d.left() / scale),
                               right=int(d.right() / scale)) for d in dets]
