import bz2
import os.path
import urllib
from os.path import expanduser

import cv2
import dlib
import numpy as np
import recognisers as rp
import rospy
# -------------------------- set gpu using tf ---------------------------
import tensorflow as tf
from cv_bridge import CvBridge
# -------------------  start importing keras module ---------------------
from keras.models import model_from_json

graph = tf.get_default_graph()


class EyeStateRecogniser(object):
    EYESTATE_JSON_FILE = expanduser("~/.dlib/eyestate.json")
    EYESTATE_MODEL_FILE = expanduser("~/.dlib/eyestate.h5")

    EYESTATE_JSON_URL = "https://raw.githubusercontent.com/mitiku1/EyeStateDetection/master/models/model.json"
    EYESTATE_MODEL_URL = "https://raw.githubusercontent.com/mitiku1/EyeStateDetection/master/models/model.h5"

    DLIB_SHAPE_MODEL_FILE = expanduser("~/.dlib/shape_predictor.dat")
    DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    THRESH_HOLD = 0.5
    IMG_SIZE = (100, 100)

    def __init__(self):
        self.models_initialized = False
        self.bridge = CvBridge()

    def initialize_models(self):
        # download and unzip models
        url_opener = urllib.URLopener()
        if not os.path.exists(expanduser("~/.dlib")):
            os.makedirs(expanduser("~/.dlib"))

        if not os.path.isfile(EyeStateRecogniser.DLIB_SHAPE_MODEL_FILE):
            rospy.loginfo("downloading %s" % EyeStateRecogniser.DLIB_SHAPE_MODEL_URL)
            url_opener.retrieve(EyeStateRecogniser.DLIB_SHAPE_MODEL_URL, EyeStateRecogniser.DLIB_SHAPE_MODEL_FILE)
            data = bz2.BZ2File(EyeStateRecogniser.DLIB_SHAPE_MODEL_FILE).read()  # get the decompressed data
            open(EyeStateRecogniser.DLIB_SHAPE_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

        if not os.path.isfile(EyeStateRecogniser.EYESTATE_JSON_FILE):
            rospy.loginfo("downloading %s" % EyeStateRecogniser.EYESTATE_JSON_URL)
            url_opener.retrieve(EyeStateRecogniser.EYESTATE_JSON_URL, EyeStateRecogniser.EYESTATE_JSON_FILE)

        if not os.path.isfile(EyeStateRecogniser.EYESTATE_MODEL_FILE):
            rospy.loginfo("downloading %s" % EyeStateRecogniser.EYESTATE_MODEL_URL)
            url_opener.retrieve(EyeStateRecogniser.EYESTATE_MODEL_URL, EyeStateRecogniser.EYESTATE_MODEL_FILE)

        # initialize predictor and model
        self.predictor = dlib.shape_predictor(EyeStateRecogniser.DLIB_SHAPE_MODEL_FILE)

        with open(EyeStateRecogniser.EYESTATE_JSON_FILE) as model_file:
            self.model = model_from_json(model_file.read())
            self.model.load_weights(EyeStateRecogniser.EYESTATE_MODEL_FILE)

        self.models_initialized = True

    @staticmethod
    def shape_to_dlib_points(shapes, original_shape, resize_shape):
        """Extracts dlib key points from face image
        parameters
        ----------
        shapes

        Returns
        -------
        numpy.ndarray
            dlib key points of the face inside rectangle.
        """

        dlib_points = np.zeros((68, 2))
        for i, part in enumerate(shapes):
            dlib_points[i] = [part.x, part.y]

        # scale dlib_points from the scale of image they were created from to another scale
        scale_h = resize_shape[1] / float(original_shape[1])
        scale_v = resize_shape[0] / float(original_shape[0])
        scale = np.array([[scale_h, scale_v]])
        dlib_points = dlib_points * scale

        return dlib_points

    # image needs to be format
    def recognize(self, image, dlib_shapes):
        if not self.models_initialized:
            rospy.logwarn("Please call initialize_models")

        with graph.as_default():
            face_img = rp.math.sanitize(image, EyeStateRecogniser.IMG_SIZE)

            dlib_points = EyeStateRecogniser.shape_to_dlib_points(dlib_shapes, image.shape, EyeStateRecogniser.IMG_SIZE)

            left_key_points_11 = EyeStateRecogniser.get_left_key_points(dlib_points)
            right_key_points_11 = EyeStateRecogniser.get_right_key_points(dlib_points)

            shape = (24, 24, 1)
            l_i, lkp, ld, la = EyeStateRecogniser.get_attributes_wrt_local_frame(face_img, left_key_points_11, shape)
            r_i, rkp, rd, ra = EyeStateRecogniser.get_attributes_wrt_local_frame(face_img, right_key_points_11, shape)

            l_i = l_i.reshape(-1, 24, 24, 1).astype(np.float32) / 255
            r_i = r_i.reshape(-1, 24, 24, 1).astype(np.float32) / 255

            lkp = np.expand_dims(lkp, 1).astype(np.float32) / 24
            ld = np.expand_dims(ld, 1).astype(np.float32) / 24
            la = np.expand_dims(la, 1).astype(np.float32) / np.pi

            rkp = np.expand_dims(rkp, 1).astype(np.float32) / 24
            rd = np.expand_dims(rd, 1).astype(np.float32) / 24
            ra = np.expand_dims(ra, 1).astype(np.float32) / np.pi

            lkp = lkp.reshape(-1, 1, 11, 2)
            ld = ld.reshape(-1, 1, 11, 1)
            la = la.reshape(-1, 1, 11, 1)

            rkp = rkp.reshape(-1, 1, 11, 2)
            rd = rd.reshape(-1, 1, 11, 1)
            ra = ra.reshape(-1, 1, 11, 1)

            left_prediction = self.model.predict([l_i, lkp, ld, la])[0]
            right_prediction = self.model.predict([r_i, rkp, rd, ra])[0]

            left_closed = ((-left_prediction[1] + left_prediction[0]) / 2.0) + 0.5
            right_closed = ((-right_prediction[1] + right_prediction[0]) / 2.0) + 0.5

        return [left_closed, right_closed]

    @staticmethod
    def get_right_key_points(key_points):
        """Extract dlib key points from right eye region including eye brow region.
        Parameters
        ----------
        key_points : numpy.ndarray
            Dlib face key points
        Returns:
            dlib key points of right eye region
        """

        output = np.zeros((11, 2))
        output[0:5] = key_points[17:22]
        output[5:11] = key_points[36:42]
        return output

    @staticmethod
    def get_left_key_points(key_points):
        """Extract dlib key points from left eye region including eye brow region.
        Parameters
        ----------
        key_points : numpy.ndarray
            Dlib face key points
        Returns:
            dlib key points of left eye region
        """
        output = np.zeros((11, 2))
        output[0:5] = key_points[22:27]
        output[5:11] = key_points[42:48]
        return output

    @staticmethod
    def get_attributes_wrt_local_frame(face_image, key_points_11, image_shape):
        """Extracts eye image, key points of the eye region with respect
        face eye image, angles and distances between centroid of key point of eye  and
        other key points of the eye.
        Parameters
        ----------
        face_image : numpy.ndarray
            Image of the face
        key_points_11 : numpy.ndarray
            Eleven key points of the eye including eyebrow region.
        image_shape : tuple
            Shape of the output eye image

        Returns
        -------
        eye_image : numpy.ndarray
            Image of the eye region
        key_points_11 : numpy.ndarray
            Eleven key points translated to eye image frame
        dists : numpy.ndarray
            Distances of each 11 key points from centeroid of all 11 key points
        angles : numpy.ndarray
            Angles between each 11 key points from centeroid

        """

        face_image_shape = face_image.shape
        top_left = key_points_11.min(axis=0)
        bottom_right = key_points_11.max(axis=0)

        # bound the coordinate system inside eye image
        bottom_right[0] = min(face_image_shape[1], bottom_right[0])
        bottom_right[1] = min(face_image_shape[0], bottom_right[1] + 5)
        top_left[0] = max(0, top_left[0])
        top_left[1] = max(0, top_left[1])

        # crop the eye
        top_left = top_left.astype(np.uint8)
        bottom_right = bottom_right.astype(np.uint8)
        eye_image = face_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # translate the eye key points from face image frame to eye image frame
        key_points_11 = key_points_11 - top_left
        key_points_11 += np.finfo(float).eps
        # horizontal scale to resize image
        scale_h = image_shape[1] / float(eye_image.shape[1])
        # vertical scale to resize image
        scale_v = image_shape[0] / float(eye_image.shape[0])

        # resize left eye image to network input size
        eye_image = cv2.resize(eye_image, (image_shape[0], image_shape[1]))

        # scale left key points proportional with respect to left eye image resize scale
        scale = np.array([[scale_h, scale_v]])
        key_points_11 = key_points_11 * scale

        # calculate centroid of left eye key points
        centroid = np.array([key_points_11.mean(axis=0)])

        # calculate distances from  centroid to each left eye key points
        dists = rp.math.distance_between(key_points_11, centroid)

        # calculate angles between centroid point vector and left eye key points vectors
        angles = rp.math.angles_between_point_vectors(key_points_11, centroid)
        return eye_image, key_points_11, dists, angles
