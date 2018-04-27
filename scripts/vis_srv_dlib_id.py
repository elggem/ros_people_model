#!/usr/bin/python
import bz2
import os
import pickle
import time
import urllib
import uuid
from os.path import expanduser

import dlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from ros_peoplemodel.srv import DlibFaceID
from ros_peoplemodel.srv import DlibFaceIDResponse

DLIB_RECOGNITION_MODEL_FILE = expanduser("~/.dlib/recognition_resnet.dat")
DLIB_RECOGNITION_MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

FACE_ID_VECTOR_FILE = expanduser("~/.dlib/faces.pkl")
FACE_ID_VECTOR_DICT = None

current_milli_time = lambda: int(round(time.time() * 1000))


def initializeModel():
    urlOpener = urllib.URLopener()
    if not os.path.exists(expanduser("~/.dlib")):
        os.makedirs(expanduser("~/.dlib"))

    if not os.path.isfile(DLIB_RECOGNITION_MODEL_FILE):
        print("downloading %s" % DLIB_RECOGNITION_MODEL_URL)
        urlOpener.retrieve(DLIB_RECOGNITION_MODEL_URL, DLIB_RECOGNITION_MODEL_FILE)
        data = bz2.BZ2File(DLIB_RECOGNITION_MODEL_FILE).read()  # get the decompressed data
        open(DLIB_RECOGNITION_MODEL_FILE, 'wb').write(data)  # write a uncompressed file


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

    # for identifier, stored in FACE_ID_VECTOR_DICT.iteritems():
    #    print stored['position']

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
        print("spatial %.4f" % (spatialdistance))
        print("time %.4f" % (timedistance))
        if spatialdistance < 100.0 and timedistance < 200:
            print("adding new vector to face")
            addFaceVectorToID(identifier, face_vec)
            persistFaceID()
            return identifier

    FACE_ID_VECTOR_DICT[uuid.uuid4().hex] = {'vector': [face_vec], 'position': np.array(position),
                                             'timestamp': timestamp}
    persistFaceID()


def handleRequest(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    d = dlib.rectangle(0, 0, image.shape[0], image.shape[1])

    points = dlib.points()
    [points.append(dlib.point(int(p.x), int(p.y))) for p in req.shape]
    dlib_shape = dlib.full_object_detection(d, points)

    # Get the face descriptor
    face_descriptor = dlib_face_recognizer.compute_face_descriptor(image, dlib_shape)
    face_id = getFaceID(face_descriptor, [req.roi.x_offset, req.roi.y_offset, req.roi.height, req.roi.width],
                        current_milli_time())

    if face_id is None:
        face_id = ""

    return DlibFaceIDResponse(face_id)


if __name__ == "__main__":
    initializeModel()
    initializeFaceID()
    bridge = CvBridge()
    dlib_face_recognizer = dlib.face_recognition_model_v1(DLIB_RECOGNITION_MODEL_FILE)

    rospy.init_node('vis_srv_dlib_id_server', anonymous=True)
    srv = rospy.Service('vis_srv_dlib_id', DlibFaceID, handleRequest)

    rospy.spin()
