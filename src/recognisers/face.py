import cv2
import dlib
import rospy
from recognisers.recogniser import Recogniser


class FaceRecogniser(Recogniser):
    DLIB_CNN_MODEL_FILE = "mmod_cnn.dat"
    DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"

    def __init__(self):
        Recogniser.__init__(self)

    def initialise(self, download=True):
        # download models
        if download:
            self.download_model(FaceRecogniser.DLIB_CNN_MODEL_URL, FaceRecogniser.DLIB_CNN_MODEL_FILE)

        if self.wait_for_model(FaceRecogniser.DLIB_CNN_MODEL_FILE):
            # Dlib detectors
            self.dlib_face_detector = dlib.cnn_face_detection_model_v1(self.get_file_path(FaceRecogniser.DLIB_CNN_MODEL_FILE))
            self.dlib_frontal_face_detector = dlib.get_frontal_face_detector()

            self.is_initialised = True

    def detect_faces(self, image, scale=1.0):
        if not self.is_initialised:
            rospy.logwarn("Please call initialise")

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
        if not self.is_initialised:
            rospy.logwarn("Please call initialise")

        if scale is not 1.0:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # perform CNN detection
        dets = self.dlib_frontal_face_detector(image, 1)

        # rescale
        return [dlib.rectangle(top=int(d.top() / scale),
                               bottom=int(d.bottom() / scale),
                               left=int(d.left() / scale),
                               right=int(d.right() / scale)) for d in dets]
