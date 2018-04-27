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
from ros_peoplemodel.srv import iCogEmopy
from ros_peoplemodel.srv import iCogEmopyResponse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# -------------------  start importing keras module ---------------------
from keras.models import model_from_json

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


def current_time_milliseconds():
    return int(round(time.time() * 1000))


def initialize_models():
    url_opener = urllib.URLopener()
    if not os.path.exists(expanduser("~/.dlib")):
        os.makedirs(expanduser("~/.dlib"))

    if not os.path.isfile(DLIB_SHAPE_MODEL_FILE):
        print("downloading %s" % DLIB_SHAPE_MODEL_URL)
        url_opener.retrieve(DLIB_SHAPE_MODEL_URL, DLIB_SHAPE_MODEL_FILE)
        data = bz2.BZ2File(DLIB_SHAPE_MODEL_FILE).read()  # get the decompressed data
        open(DLIB_SHAPE_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

    if not os.path.isfile(EMOPY_AVA_JSON_FILE):
        print("downloading %s" % EMOPY_AVA_JSON_URL)
        url_opener.retrieve(EMOPY_AVA_JSON_URL, EMOPY_AVA_JSON_FILE)

    if not os.path.isfile(EMOPY_AVA_MODEL_FILE):
        print("downloading %s" % EMOPY_AVA_MODEL_URL)
        url_opener.retrieve(EMOPY_AVA_MODEL_URL, EMOPY_AVA_MODEL_FILE)


def sanitize(image):
    """
        Converts image into gray scale if it RGB image and resize it to IMG_SIZE

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            gray scale image resized to IMG_SIZE
        """
    if image is None:
        return None
    assert len(image.shape) == 2 or len(image.shape) == 3, "Image dim should be either 2 or 3. It is " + str(
        len(image.shape))

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    return image


def get_dlib_points(image, predictor):
    """
    Get dlib facial key points of face

    Parameters
    ----------
    image : numpy.ndarray
        face image.
    Returns
    -------
    numpy.ndarray
        68 facial key points
    """
    face = dlib.rectangle(0, 0, image.shape[1] - 1, image.shape[0] - 1)
    img = image.reshape(IMG_SIZE[0], IMG_SIZE[1])
    shapes = predictor(img, face)
    parts = shapes.parts()
    output = np.zeros((68, 2))
    for i, point in enumerate(parts):
        output[i] = [point.x, point.y]
    output = np.array(output).reshape((1, 68, 2))
    return output


def to_dlib_points(images, predictor):
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
    for i in range(len(images)):
        dlib_points = get_dlib_points(images[i], predictor)[0]
        centroid = np.mean(dlib_points, axis=0)
        centroids[i] = centroid
        output[i][0] = dlib_points
    return output, centroids


def get_distances_angles(all_dlib_points, centroids):
    """
    Get the distances for each dlib facial key points in face from centroid of the points and
    angles between the dlib points vector and centroid vector.

    Parameters
    ----------
    all_dlib_points : numpy.ndarray
        dlib facial key points for each face.
    centroid :
        centroid of dlib facial key point for each face
    Returns
    -------
    numpy.ndarray , numpy.ndarray
        Dlib landmarks distances and angles with respect to respective centroid.
    """
    all_distances = np.zeros((len(all_dlib_points), 1, 68, 1))
    all_angles = np.zeros((len(all_dlib_points), 1, 68, 1))
    for i in range(len(all_dlib_points)):
        dists = np.linalg.norm(centroids[i] - all_dlib_points[i][0], axis=1)
        angles = get_angles(all_dlib_points[i][0], centroids[i])
        all_distances[i][0] = dists.reshape(1, 68, 1)
        all_angles[i][0] = angles.reshape(1, 68, 1)
    return all_distances, all_angles


def angle_between(p1, p2):
    """
    Get clockwise angle between two vectors

    Parameters
    ----------
    p1 : numpy.ndarray
        first vector.
    p2 : numpy.ndarray
        second vector.
    Returns
    -------
    float
        angle in radiuns
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)


def get_angles(dlib_points, centroid):
    """
    Get clockwise angles between dlib landmarks of face and centroid of landmarks.

    Parameters
    ----------
    dlib_points : numpy.ndarray
        dlib landmarks of face.
    centroid : numpy.ndarray
        centroid of dlib landrmask.
    Returns
    -------
    numpy.ndarray
        dlib points clockwise angles in radiuns with respect to centroid vector
    """
    output = np.zeros((68))
    for i in range(68):
        angle = angle_between(dlib_points[i], centroid)
        output[i] = angle
    return output


def arg_max(array):
    """
    Get index of maximum element of 1D array

    Parameters
    ----------
    array : list

    Returns
    -------
    int
        index of maximum element of the array
    """
    max_value = array[0]
    max_index = 0
    for i, el in enumerate(array):
        if max_value < el:
            max_value = el
            max_index = i
    return max_index


def recognize_emotion(model, predictor, face, model_type="ava"):
    """
    Recognize emotion single face image.

    Parameters
    ----------
    model : keras.models.Model
        model used to predict emotion.
    image : numpy.ndarray
        face image.
    Returns
    -------
    str, int
        emotion and length of outputs of model.
    """
    face = sanitize(face)
    face = face.reshape(-1, 48, 48, 1)
    if model_type != "ava-ii":
        dlibpoints, centroids = to_dlib_points(face, predictor)
        dists, angles = get_distances_angles(dlibpoints, centroids)
        dlibpoints = dlibpoints.astype(float) / 50
        dists = dists.astype(float) / 50
        angles = angles.astype(float) / 50
        face = face.reshape(face.shape[0], 48, 48, 1)
        face = face.astype('float32')
        face /= 255
        predictions = model.predict([face, dlibpoints, dists, angles])[0]
    else:
        predictions = model.predict(face)[0]
    return predictions


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    with graph.as_default():
        emotions = recognize_emotion(emopy_model, dlib_shape_predictor, image)
    return iCogEmopyResponse(emotions)


if __name__ == "__main__":
    initialize_models()
    bridge = CvBridge()

    dlib_shape_predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)

    # emopy
    with open(EMOPY_AVA_JSON_FILE) as model_file:
        emopy_model = model_from_json(model_file.read())
        emopy_model.load_weights(EMOPY_AVA_MODEL_FILE)

    graph = tf.get_default_graph()

    rospy.init_node('vis_srv_icog_emopy_server', anonymous=True)
    srv = rospy.Service('vis_srv_icog_emopy', iCogEmopy, handle_request)

    rospy.spin()
