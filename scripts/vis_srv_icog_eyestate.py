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
from ros_peoplemodel.srv import iCogEyeState
from ros_peoplemodel.srv import iCogEyeStateResponse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# -------------------  start importing keras module ---------------------
from keras.models import model_from_json

EYESTATE_JSON_FILE = expanduser("~/.dlib/eyestate.json")
EYESTATE_MODEL_FILE = expanduser("~/.dlib/eyestate.h5")

EYESTATE_JSON_URL = "https://raw.githubusercontent.com/mitiku1/EyeStateDetection/master/models/model.json"
EYESTATE_MODEL_URL = "https://raw.githubusercontent.com/mitiku1/EyeStateDetection/master/models/model.h5"

DLIB_SHAPE_MODEL_FILE = expanduser("~/.dlib/shape_predictor.dat")
DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

THRESH_HOLD = 0.5
IMG_SIZE = (48, 48)


def current_time_milliseconds():
    return int(round(time.time() * 1000))


def initialize_models():
    url_opener = urllib.URLopener()
    if not os.path.exists(expanduser("~/.dlib")):
        os.makedirs(expanduser("~/.dlib"))

    if not os.path.isfile(DLIB_SHAPE_MODEL_FILE):
        rospy.loginfo("downloading %s" % DLIB_SHAPE_MODEL_URL)
        url_opener.retrieve(DLIB_SHAPE_MODEL_URL, DLIB_SHAPE_MODEL_FILE)
        data = bz2.BZ2File(DLIB_SHAPE_MODEL_FILE).read()  # get the decompressed data
        open(DLIB_SHAPE_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

    if not os.path.isfile(EYESTATE_JSON_FILE):
        rospy.loginfo("downloading %s" % EYESTATE_JSON_URL)
        url_opener.retrieve(EYESTATE_JSON_URL, EYESTATE_JSON_FILE)

    if not os.path.isfile(EYESTATE_MODEL_FILE):
        rospy.loginfo("downloading %s" % EYESTATE_MODEL_URL)
        url_opener.retrieve(EYESTATE_MODEL_URL, EYESTATE_MODEL_FILE)


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


def get_dlib_points(img, predictor, rectangle):
    """Extracts dlib key points from face image
    parameters
    ----------
    img : numpy.ndarray
        Grayscale face image
    predictor : dlib.shape_predictor
        shape predictor which is used to localize key points from face image
    rectangle : dlib.rectangle
        face bounding box inside image
    Returns
    -------
    numpy.ndarray
        dlib key points of the face inside rectangle.
    """

    shape = predictor(img, rectangle)
    dlib_points = np.zeros((68, 2))
    for i, part in enumerate(shape.parts()):
        dlib_points[i] = [part.x, part.y]
    return dlib_points


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


def distance_between(v1, v2):
    """Calculates euclidean distance between two vectors.
    If one of the arguments is matrix then the output is calculated for each row
    of that matrix.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Matrix if one of the arguments is matrix and vector if both arguments are vectors.
    """

    diff = v2 - v1
    diff_squared = np.square(diff)
    dist_squared = diff_squared.sum(axis=1)
    dists = np.sqrt(dist_squared)
    return dists


def angles_between(v1, v2):
    """Calculates angle between two point vectors.
    Parameters
    ----------
    v1 : numpy.ndarray
        First vector
    v2 : numpy.ndarray
        Second vector

    Returns:
    --------
    numpy.ndarray
        Vector if one of the arguments is matrix and scalar if both arguments are vectors.
    """
    dot_prod = (v1 * v2).sum(axis=1)
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)

    cosine_of_angle = (dot_prod / (v1_norm * v2_norm)).reshape(11, 1)

    angles = np.arccos(np.clip(cosine_of_angle, -1, 1))

    return angles


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
    dists = distance_between(key_points_11, centroid)

    # calculate angles between centroid point vector and left eye key points vectors
    angles = angles_between(key_points_11, centroid)
    return eye_image, key_points_11, dists, angles


def get_left_eye_attributes(face_image, predictor, image_shape):
    """Extracts eye image, key points, distance of each key points
    from centroid of the key points and angles between centroid and
    each key points of left eye.

    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    predictor : dlib.shape_predictor
        Dlib Shape predictor to extract key points
    image_shape : tuple
        The output eye image shape
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
    face_rect = dlib.rectangle(0, 0, face_image_shape[1], face_image_shape[0])
    kps = get_dlib_points(face_image, predictor, face_rect)
    # Get key points of the eye and eyebrow

    key_points_11 = get_left_key_points(kps)

    eye_image, key_points_11, dists, angles = get_attributes_wrt_local_frame(face_image, key_points_11, image_shape)

    return eye_image, key_points_11, dists, angles


def get_right_eye_attributes(face_image, predictor, image_shape):
    """Extracts eye image, key points, distance of each key points
    from centroid of the key points and angles between centroid and
    each key points of right eye.

    Parameters
    ----------
    face_image : numpy.ndarray
        Image of the face
    predictor : dlib.shape_predictor
        Dlib Shape predictor to extract key points
    image_shape : tuple
        The output eye image shape
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
    face_rect = dlib.rectangle(0, 0, face_image_shape[1], face_image_shape[0])
    kps = get_dlib_points(face_image, predictor, face_rect)
    # Get key points of the eye and eyebrow

    key_points_11 = get_right_key_points(kps)

    eye_image, key_points_11, dists, angles = get_attributes_wrt_local_frame(face_image, key_points_11, image_shape)

    return eye_image, key_points_11, dists, angles


def recognize_eyestate(image):
    face_img = sanitize(image)
    # cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),color=(255,0,0),thickness=2)
    face_img = cv2.resize(face_img, (100, 100))
    l_i, lkp, ld, la = get_left_eye_attributes(face_img, predictor, (24, 24, 1))
    r_i, rkp, rd, ra = get_right_eye_attributes(face_img, predictor, (24, 24, 1))

    # cv2.imshow("Left eye: ",l_i)
    # for kp in lkp:
    #     cv2.circle(l_i,(kp[0],kp[1]),1,(255,255,0))
    # cv2.imshow("Right eye: ",r_i)
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

    left_prediction = model.predict([l_i, lkp, ld, la])[0]
    right_prediction = model.predict([r_i, rkp, rd, ra])[0]

    left_arg_max = np.argmax(left_prediction)
    right_arg_max = np.argmax(right_prediction)

    left_closed = ((-left_prediction[1] + left_prediction[0]) / 2.0) + 0.5
    right_closed = ((-right_prediction[1] + right_prediction[0]) / 2.0) + 0.5

    return iCogEyeStateResponse([left_closed, right_closed])


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    with graph.as_default():
        eyes_closed = recognize_eyestate(image)
    return eyes_closed


if __name__ == "__main__":
    initialize_models()
    bridge = CvBridge()

    predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)

    # emopy
    with open(EYESTATE_JSON_FILE) as model_file:
        model = model_from_json(model_file.read())
        model.load_weights(EYESTATE_MODEL_FILE)

    graph = tf.get_default_graph()

    rospy.init_node('vis_srv_icog_eyestate_server', anonymous=True)
    srv = rospy.Service('vis_srv_icog_eyestate', iCogEyeState, handle_request)

    rospy.spin()
