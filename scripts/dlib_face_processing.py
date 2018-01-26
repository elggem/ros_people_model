#!/usr/bin/python
import sys
import dlib
from skimage import io

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ros_slopp.msg import Face

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import os.path
import urllib

import pickle
import uuid

import bz2
import cv2

"""
This script uses the various detectors in DLib to do
  1 Face detection using CNN detector to do face detection for frontal and sideways faces
    -> can we look at confidence to determine if frontal?
  2 For each detected face it performs detection to detect frontal faces
  3 For each frontal face it outputs the dected shape
  4 For each frontal face it performs 128D vector calculation and matches it with an internal array




TODO:
  - detect if shapekeys are bad
  - improve recognition (clustering?)

  - gaze direction
  https://github.com/severin-lemaignan/gazr

  - parametrize various features
  - output FPS to some topic

"""




DLIB_CNN_MODEL_FILE = "/tmp/dlib/mmod_cnn.dat"
DLIB_SHAPE_MODEL_FILE = "/tmp/dlib/shape_predictor.dat"
DLIB_RECOGNITION_MODEL_FILE = "/tmp/dlib/recognition_resnet.dat"

DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
DLIB_SHAPE_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_RECOGNITION_MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

def initializeModels():
    urlOpener = urllib.URLopener()
    if not os.path.exists("/tmp/dlib"):
        os.makedirs("/tmp/dlib")

    if not os.path.isfile(DLIB_CNN_MODEL_FILE):
        print("downloading %s" % DLIB_CNN_MODEL_URL)
        urlOpener.retrieve(DLIB_CNN_MODEL_URL, DLIB_CNN_MODEL_FILE)
        data = bz2.BZ2File(DLIB_CNN_MODEL_FILE).read() # get the decompressed data
        open(DLIB_CNN_MODEL_FILE, 'wb').write(data) # write a uncompressed file

    if not os.path.isfile(DLIB_SHAPE_MODEL_FILE):
        print("downloading %s" % DLIB_SHAPE_MODEL_URL)
        urlOpener.retrieve(DLIB_SHAPE_MODEL_URL, DLIB_SHAPE_MODEL_FILE)
        data = bz2.BZ2File(DLIB_SHAPE_MODEL_FILE).read() # get the decompressed data
        open(DLIB_SHAPE_MODEL_FILE, 'wb').write(data) # write a uncompressed file

    if not os.path.isfile(DLIB_RECOGNITION_MODEL_FILE):
        print("downloading %s" % DLIB_RECOGNITION_MODEL_URL)
        urlOpener.retrieve(DLIB_RECOGNITION_MODEL_URL, DLIB_RECOGNITION_MODEL_FILE)
        data = bz2.BZ2File(DLIB_RECOGNITION_MODEL_FILE).read() # get the decompressed data
        open(DLIB_RECOGNITION_MODEL_FILE, 'wb').write(data) # write a uncompressed file






FACE_ID_VECTOR_FILE = "/tmp/faces.pkl"
FACE_ID_VECTOR_DICT = None

def initializeFaceID():
    # Determines if there is pickled face recognition array on disk and restores it.
    # Otherwise initializes empty array.
    global FACE_ID_VECTOR_DICT
    if FACE_ID_VECTOR_DICT is None:
        if os.path.isfile(FACE_ID_VECTOR_FILE):
            print("Loading Face ID data")
            FACE_ID_VECTOR_DICT = pickle.load(open(FACE_ID_VECTOR_FILE, "r"))
        else:
            FACE_ID_VECTOR_DICT = {}

def persistFaceID():
    print("Persisting Face ID data")
    pickle.dump(FACE_ID_VECTOR_DICT, open(FACE_ID_VECTOR_FILE, "w"))

def getFaceID(face_vec, threshold=0.6):
    # Compares given face vector with stored to determine match
    for identifier, stored_vec in FACE_ID_VECTOR_DICT.iteritems():
        if np.linalg.norm(np.array(face_vec) - stored_vec) < threshold:
            return identifier
    # TODO: Maybe add new face only on threshold.
    FACE_ID_VECTOR_DICT[uuid.uuid4().hex] = np.array(face_vec)
    persistFaceID()

def performCNNFaceDetection(img, scale=1.0):
    if scale is not 1.0:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # perform CNN detection
    cnnDets = dlib_cnn_detector(img, 1)
    # rescale
    return [dlib.rectangle(top    = int(d.rect.top()    / scale),
                           bottom = int(d.rect.bottom() / scale),
                           left   = int(d.rect.left()   / scale),
                           right  = int(d.rect.right()  / scale)) for d in cnnDets]

def performFaceDetection(img, scale=1.0):
    if scale is not 1.0:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # perform CNN detection
    dets = dlib_detector(img, 1)
    # rescale
    return [dlib.rectangle(top    = int(d.top()    / scale),
                           bottom = int(d.bottom() / scale),
                           left   = int(d.left()   / scale),
                           right  = int(d.right()  / scale)) for d in dets]


IMAGE = None
FACE_CANDIDATES_CNN = []
FACE_CANDIDATES_FRONTAL = []

def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "rgb8")


def faceDetectCNNCallback(event):
    global IMAGE, FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = performCNNFaceDetection(IMAGE, scale=0.75)


def faceDetectFrontalCallback(event):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL

    frontal_dets = []
    #goes through list and only saves the one
    for k, d in enumerate(FACE_CANDIDATES_CNN):
        padding = int(IMAGE.shape[0]*0.1)
        t = np.maximum(d.top()  - padding, 0)
        l = np.maximum(d.left() - padding, 0)
        b = np.minimum(d.bottom() + padding, IMAGE.shape[0])
        r = np.minimum(d.right()  + padding, IMAGE.shape[1])
        cropped_face = IMAGE[t:b, l:r, :]

        dets = performFaceDetection(cropped_face, scale=0.25)

        if len(dets)==1:
            frontal_dets.append(dlib.rectangle(   top = t + dets[0].top(),
                                               bottom = t + dets[0].bottom(),
                                                 left = l + dets[0].left(),
                                                right = l + dets[0].right()))

    FACE_CANDIDATES_FRONTAL = frontal_dets


def faceAnalysis(event):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL


    if IMAGE is not None:
        win.set_image(IMAGE)



    for k, d in enumerate(FACE_CANDIDATES_FRONTAL):
        face = Face()
        #face.image = bridge.cv2_to_imgmsg(np.array(cropped_face))
        face.bounding_box = [d.top(), d.bottom(), d.left(), d.right()]

        # Get the shape
        shape = dlib_shape_predictor(IMAGE, d)
        face.shape = [Point(p.x, p.y, 0) for p in shape.parts()]

        # Get the face descriptor
        face_descriptor = dlib_face_recognizer.compute_face_descriptor(IMAGE, shape)
        face.face_id = getFaceID(face_descriptor)

        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        pub.publish(face)
    for d in FACE_CANDIDATES_CNN:
        win.add_overlay(d)

if __name__ == "__main__":
    initializeModels()
    initializeFaceID()

    rospy.init_node('dlib_node', anonymous=True)

    win = dlib.image_window()
    bridge = CvBridge()

    # Publishers
    pub = rospy.Publisher('faces', Face, queue_size=10)
    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_cnn_detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_MODEL_FILE)
    dlib_shape_predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)
    dlib_face_recognizer = dlib.face_recognition_model_v1(DLIB_RECOGNITION_MODEL_FILE)

    # Launch detectors
    rospy.Timer(rospy.Duration(1.0/5.0), faceDetectCNNCallback)
    rospy.Timer(rospy.Duration(1.0/5.0), faceDetectFrontalCallback)
    rospy.Timer(rospy.Duration(1.0/30.0), faceAnalysis)

    rospy.spin()
