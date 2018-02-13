#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from ros_slopp.msg import Face

DLIB_RECOGNITION_MODEL_FILE = "/tmp/dlib/recognition_resnet.dat"
DLIB_RECOGNITION_MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

FACE_ID_VECTOR_FILE = "/tmp/faces.pkl"
FACE_ID_VECTOR_DICT = None

ANALYSIS_FRATE = 1.0/30.0

def initializeModels():
    urlOpener = urllib.URLopener()
    if not os.path.exists("/tmp/dlib"):
        os.makedirs("/tmp/dlib")

    if not os.path.isfile(DLIB_RECOGNITION_MODEL_FILE):
        print("downloading %s" % DLIB_RECOGNITION_MODEL_URL)
        urlOpener.retrieve(DLIB_RECOGNITION_MODEL_URL, DLIB_RECOGNITION_MODEL_FILE)
        data = bz2.BZ2File(DLIB_RECOGNITION_MODEL_FILE).read() # get the decompressed data
        open(DLIB_RECOGNITION_MODEL_FILE, 'wb').write(data) # write a uncompressed file



def initializeFaceID():
    # Determines if there is pickled face recognition array on disk and restores it.
    # Otherwise initializes empty array.
    global FACE_ID_VECTOR_DICT
    if FACE_ID_VECTOR_DICT is None:
        if os.path.isfile(FACE_ID_VECTOR_FILE):
            print("Loading Face ID data")
            FACE_ID_VECTOR_DICT = pickle.load(open(FACE_ID_VECTOR_FILE, "rb"))
            print("number of faces registered: %i" % len(FACE_ID_VECTOR_DICT))
        else:
            FACE_ID_VECTOR_DICT = {}

def persistFaceID():
    print("Persisting Face ID data")
    pickle.dump(FACE_ID_VECTOR_DICT, open(FACE_ID_VECTOR_FILE, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def addFaceVectorToID(identifier, face_vec):
    old_vectors = FACE_ID_VECTOR_DICT[identifier]['vector']
    FACE_ID_VECTOR_DICT[identifier]['vector'] = old_vectors + [face_vec]


def getFaceID(face_vec, position, timestamp, threshold=0.6):
    face_vec = np.array([i for i in face_vec])

    # Compares given face vector with stored to determine match
    for identifier, stored in FACE_ID_VECTOR_DICT.iteritems():
        for stored_vector in stored['vector']:
            if np.linalg.norm(face_vec - stored_vector) < threshold:
                FACE_ID_VECTOR_DICT[identifier]['position'] = np.array(position)
                FACE_ID_VECTOR_DICT[identifier]['timestamp'] = timestamp
                return identifier

    # check if sth close.
    for identifier, stored in FACE_ID_VECTOR_DICT.iteritems():
        timedistance = np.abs(stored['timestamp'] - np.array(timestamp))
        spatialdistance = np.linalg.norm(stored['position'] - np.array(position))
        print("times %.4f" % (spatialdistance))
        if spatialdistance < 100.0 and timedistance < 1000:
            print("adding new vector to face")
            addFaceVectorToID(identifier, face_vec)
            persistFaceID()
            return identifier


    # TODO: Maybe add new face only on threshold.
    FACE_ID_VECTOR_DICT[uuid.uuid4().hex] = {'vector': [face_vec], 'position':np.array(position), 'timestamp':timestamp}
    persistFaceID()


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")


def faceAnalysis(event):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL, FACE_CANDIDATES_SIDEWAYS

    for k, d in enumerate(FACE_CANDIDATES_FRONTAL):
        face = Face()
        cropped_face = IMAGE[d.top():d.bottom(), d.left():d.right(), :]

        # Get the face descriptor
        face_descriptor = dlib_face_recognizer.compute_face_descriptor(IMAGE, shape)
        face.face_id = getFaceID(face_descriptor, face.bounding_box, current_milli_time())

        if face.face_id is None:
            face.face_id = "            "

        ## IN CASE WE WANNA SEE IT
        faces.append(face)
        pub.publish(face)

    if DEBUG_DRAW:
        debugDraw(FACE_CANDIDATES_FRONTAL, faces)


if __name__ == "__main__":
    initializeModel()
    initializeFaceID()

    rospy.init_node('vis_dlib_id', anonymous=True)

    bridge = CvBridge()

    # Publishers
    pub = rospy.Publisher('vis_dlib_id', Attribute, queue_size=10)
    # Subscribers
    rospy.Subscriber("/vis_dlib_shapes", Image, imageCallback)

    # Dlib
    dlib_face_recognizer = dlib.face_recognition_model_v1(DLIB_RECOGNITION_MODEL_FILE)

    # Launch detectors
    rospy.Timer(rospy.Duration(ANALYSIS_FRATE), faceAnalysis)

    rospy.spin()
