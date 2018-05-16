from cv_bridge import CvBridge
from os.path import expanduser

import dlib
import rospy
from recognisers.recogniser import Recogniser


class FaceLandmarksRecogniser(Recogniser):
    DLIB_SHAPE_MODEL_FILE = "shape_predictor.dat"
    DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    def __init__(self):
        Recogniser.__init__(self)
        self.bridge = CvBridge()

    def initialise(self, download=True):
        # download models
        self.download_model(FaceLandmarksRecogniser.DLIB_SHAPE_MODEL_URL, FaceLandmarksRecogniser.DLIB_SHAPE_MODEL_FILE)

        # open models
        self.dlib_shape_predictor = dlib.shape_predictor(self.get_file_path(FaceLandmarksRecogniser.DLIB_SHAPE_MODEL_FILE))

        self.is_initialised = True

    def recognize(self, image):
        if not self.is_initialised:
            rospy.logwarn("Please call initialise")

        d = dlib.rectangle(0, 0, image.shape[0], image.shape[1])

        return self.dlib_shape_predictor(image, d)
