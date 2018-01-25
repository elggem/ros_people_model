#!/usr/bin/python
import sys
import dlib
from skimage import io

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import os.path
import urllib

import pickle
import uuid

import bz2

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

  - 

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



def analyzeImage(data):
    img = bridge.imgmsg_to_cv2(data, "rgb8")
    cnn_dets = dlib_cnn_detector(img, 1)

    win.clear_overlay()
    win.set_image(img)

    #print("CNN nr of faces detected: {}".format(len(cnn_dets)))
    rects = dlib.rectangles()
    rects.extend([d.rect for d in cnn_dets])
    win.add_overlay(rects)

    #print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(cnn_dets):
        # add some padding and generate image
        padding= int(img.shape[0]*0.2)

        top =    np.maximum(d.rect.top()   - padding, 0)
        bottom = np.minimum(d.rect.bottom()+ padding, img.shape[0])
        left =   np.maximum(d.rect.left()  - padding, 0)
        right =  np.minimum(d.rect.right() + padding, img.shape[1])

        cropped_face = img[top:bottom, left:right,0]

        dets = dlib_detector(cropped_face, 1)

        if len(dets)==1:
            shape = dlib_shape_predictor(img, d.rect)
            print shape
            win.add_overlay(shape)

            face_descriptor = dlib_face_recognizer.compute_face_descriptor(img, shape)
            face_id = getFaceID(face_descriptor)
            print face_id


    pub.publish("hhioi")






if __name__ == "__main__":
    initializeModels()
    initializeFaceID()

    rospy.init_node('dlib_node', anonymous=True)

    win = dlib.image_window()
    bridge = CvBridge()

    # Publishers
    pub = rospy.Publisher('chatter', String, queue_size=10)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_cnn_detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_MODEL_FILE)
    dlib_shape_predictor = dlib.shape_predictor(DLIB_SHAPE_MODEL_FILE)
    dlib_face_recognizer = dlib.face_recognition_model_v1(DLIB_RECOGNITION_MODEL_FILE)



    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, analyzeImage)

    rospy.spin()
