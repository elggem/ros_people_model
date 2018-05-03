#!/usr/bin/python
import bz2
import os.path
import time
import urllib
from os.path import expanduser

import cv2
import dlib
import numpy as np
import rospy
# -------------------------- set gpu using tf ---------------------------
import tensorflow as tf
from cv_bridge import CvBridge
import ros

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# -------------------  start importing keras module ---------------------
from keras.models import model_from_json
import ros_people_model as rp

graph = tf.get_default_graph()


class EmotionRecogniser(object):
    DLIB_SHAPE_MODEL_FILE = expanduser("~/.dlib/shape_predictor.dat")
    EMOPY_AVA_JSON_FILE = expanduser("~/.dlib/ava.json")
    EMOPY_AVA_MODEL_FILE = expanduser("~/.dlib/ava.h5")

    DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    EMOPY_AVA_JSON_URL = "https://raw.githubusercontent.com/mitiku1/Emopy-Models/master/models/ava.json"
    EMOPY_AVA_MODEL_URL = "https://raw.githubusercontent.com/mitiku1/Emopy-Models/master/models/ava.h5"

    EMOTION_STATES = {
        0: "neutral",
        1: "positive"
    }

    EMOTIONS = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    }

    THRESH_HOLD = 0.5
    IMG_SIZE = (48, 48)

    def __init__(self):
        self.bridge = CvBridge()

    @staticmethod
    def current_time_milliseconds():
        return int(round(time.time() * 1000))

    def initialize_models(self):
        url_opener = urllib.URLopener()
        if not os.path.exists(expanduser("~/.dlib")):
            os.makedirs(expanduser("~/.dlib"))

        if not os.path.isfile(EmotionRecogniser.DLIB_SHAPE_MODEL_FILE):
            rospy.loginfo("downloading %s" % EmotionRecogniser.DLIB_SHAPE_MODEL_URL)
            url_opener.retrieve(EmotionRecogniser.DLIB_SHAPE_MODEL_URL, EmotionRecogniser.DLIB_SHAPE_MODEL_FILE)
            data = bz2.BZ2File(EmotionRecogniser.DLIB_SHAPE_MODEL_FILE).read()  # get the decompressed data
            open(EmotionRecogniser.DLIB_SHAPE_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

        if not os.path.isfile(EmotionRecogniser.EMOPY_AVA_JSON_FILE):
            rospy.loginfo("downloading %s" % EmotionRecogniser.EMOPY_AVA_JSON_URL)
            url_opener.retrieve(EmotionRecogniser.EMOPY_AVA_JSON_URL, EmotionRecogniser.EMOPY_AVA_JSON_FILE)

        if not os.path.isfile(EmotionRecogniser.EMOPY_AVA_MODEL_FILE):
            rospy.loginfo("downloading %s" % EmotionRecogniser.EMOPY_AVA_MODEL_URL)
            url_opener.retrieve(EmotionRecogniser.EMOPY_AVA_MODEL_URL, EmotionRecogniser.EMOPY_AVA_MODEL_FILE)

        self.dlib_shape_predictor = dlib.shape_predictor(EmotionRecogniser.DLIB_SHAPE_MODEL_FILE)

        # emopy
        with open(EmotionRecogniser.EMOPY_AVA_JSON_FILE) as model_file:
            self.model = model_from_json(model_file.read())
            self.model.load_weights(EmotionRecogniser.EMOPY_AVA_MODEL_FILE)

    def recognize(self, image, dlib_shapes, model_type="ava"):
        """
        Recognize emotion single face image.

        Parameters
        ----------
        model : keras.models.Model
            model used to predict emotion.
        image : numpy.ndarray
            face image.
        face_landmarks:
            dlib 68 face landmarks
        Returns
        -------
        str, int
            emotion and length of outputs of model.
        """

        with graph.as_default():
            face = rp.math.sanitize(image, EmotionRecogniser.IMG_SIZE)
            face = face.reshape(-1, 48, 48, 1)
            if model_type != "ava-ii":
                dlibpoints, centroids = EmotionRecogniser.to_dlib_points(face, dlib_shapes, image.shape, EmotionRecogniser.IMG_SIZE) # EmotionRecogniser.to_dlib_points(face, self.dlib_shape_predictor)
                dists, angles = rp.math.get_distances_angles(dlibpoints, centroids)
                dlibpoints = dlibpoints.astype(float) / 50
                dists = dists.astype(float) / 50
                angles = angles.astype(float) / 50
                face = face.reshape(face.shape[0], 48, 48, 1)
                face = face.astype('float32')
                face /= 255
                predictions = self.model.predict([face, dlibpoints, dists, angles])[0]
            else:
                predictions = self.model.predict(face)[0]

        return predictions

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
        for i, part in enumerate(shapes.parts()):
            dlib_points[i] = [part.x, part.y]

        # scale dlib_points from the scale of image they were created from to another scale
        scale_h = resize_shape[1] / float(original_shape[1])
        scale_v = resize_shape[0] / float(original_shape[0])
        scale = np.array([[scale_h, scale_v]])
        dlib_points = dlib_points * scale

        dlib_points = np.array(dlib_points).reshape((1, 68, 2))
        return dlib_points

    @staticmethod
    def to_dlib_points(images, dlib_shapes, original_shape, resize_shape):
        """
        Get dlib facial key points of faces
        Parameters
        ----------
        images : numpy.ndarray
            faces image.
        Returns
        -------
        numpy.ndarray
            68 facial key points for each faces
        """
        output = np.zeros((len(images), 1, 68, 2))
        centroids = np.zeros((len(images), 2))
        dlib_points = EmotionRecogniser.shape_to_dlib_points(dlib_shapes, original_shape, resize_shape)
        for i in range(len(images)):
            # dlib_points =  #landmarks
            centroid = np.mean(dlib_points[0], axis=0)
            centroids[i] = centroid
            output[i][0] = dlib_points[0]
        return output, centroids

