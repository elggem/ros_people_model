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


class FaceIdRecogniser(object):
    DLIB_RECOGNITION_MODEL_FILE = expanduser("~/.dlib/recognition_resnet.dat")
    DLIB_RECOGNITION_MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

    FACE_ID_VECTOR_FILE = expanduser("~/.dlib/faces.pkl")
    FACE_ID_VECTOR_DICT = None

    def __init__(self):
        self.bridge = CvBridge()
        self.dlib_face_recognizer = dlib.face_recognition_model_v1(FaceIdRecogniser.DLIB_RECOGNITION_MODEL_FILE)

    @staticmethod
    def current_time_milliseconds():
        return int(round(time.time() * 1000))

    def initialize_models(self):
        url_opener = urllib.URLopener()
        if not os.path.exists(expanduser("~/.dlib")):
            os.makedirs(expanduser("~/.dlib"))

        if not os.path.isfile(FaceIdRecogniser.DLIB_RECOGNITION_MODEL_FILE):
            rospy.loginfo("downloading %s" % FaceIdRecogniser.DLIB_RECOGNITION_MODEL_URL)
            url_opener.retrieve(FaceIdRecogniser.DLIB_RECOGNITION_MODEL_URL,
                                FaceIdRecogniser.DLIB_RECOGNITION_MODEL_FILE)
            data = bz2.BZ2File(FaceIdRecogniser.DLIB_RECOGNITION_MODEL_FILE).read()  # get the decompressed data
            open(FaceIdRecogniser.DLIB_RECOGNITION_MODEL_FILE, 'wb').write(data)  # write a uncompressed file

        # Determines if there is pickled face recognition array on disk and restores it.
        # Otherwise initializes empty array.
        if FaceIdRecogniser.FACE_ID_VECTOR_DICT is None:
            if os.path.isfile(FaceIdRecogniser.FACE_ID_VECTOR_FILE):
                rospy.loginfo("Loading Face ID data")
                FaceIdRecogniser.FACE_ID_VECTOR_DICT = pickle.load(open(FaceIdRecogniser.FACE_ID_VECTOR_FILE, "rb"))
                rospy.loginfo("number of faces registered: %i" % len(FaceIdRecogniser.FACE_ID_VECTOR_DICT))
            else:
                FaceIdRecogniser.FACE_ID_VECTOR_DICT = {}

    def persist_face_id(self):
        rospy.logdebug("Persisting Face ID data")
        pickle.dump(FaceIdRecogniser.FACE_ID_VECTOR_DICT, open(FaceIdRecogniser.FACE_ID_VECTOR_FILE, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

    def add_face_vector_to_id(self, identifier, face_vec):
        old_vectors = FaceIdRecogniser.FACE_ID_VECTOR_DICT[identifier]['vector']
        FaceIdRecogniser.FACE_ID_VECTOR_DICT[identifier]['vector'] = old_vectors + [face_vec]

    def get_face_id(self, face_vec, position, timestamp, threshold=0.6):
        face_vec = np.array([i for i in face_vec])

        # for identifier, stored in FaceIdRecogniser.FACE_ID_VECTOR_DICT.iteritems():
        #    print stored['position']

        # Compares given face vector with stored to determine match
        for identifier, stored in FaceIdRecogniser.FACE_ID_VECTOR_DICT.iteritems():
            for stored_vector in stored['vector']:
                if np.linalg.norm(face_vec - stored_vector) < threshold:
                    FaceIdRecogniser.FACE_ID_VECTOR_DICT[identifier]['position'] = np.array(position)
                    FaceIdRecogniser.FACE_ID_VECTOR_DICT[identifier]['timestamp'] = timestamp
                    return identifier

        # check if sth close.
        for identifier, stored in FaceIdRecogniser.FACE_ID_VECTOR_DICT.iteritems():
            timedistance = np.abs(stored['timestamp'] - np.array(timestamp))
            spatialdistance = np.linalg.norm(stored['position'] - np.array(position))
            rospy.logdebug("spatial %.4f" % (spatialdistance))
            rospy.logdebug("time %.4f" % (timedistance))
            if spatialdistance < 100.0 and timedistance < 200:
                rospy.logdebug("adding new vector to face")
                self.add_face_vector_to_id(identifier, face_vec)
                self.persist_face_id()
                return identifier

        FaceIdRecogniser.FACE_ID_VECTOR_DICT[uuid.uuid4().hex] = {'vector': [face_vec], 'position': np.array(position),
                                                 'timestamp': timestamp}
        self.persist_face_id()

    def recognize(self, image, roi, landmarks):
        d = dlib.rectangle(0, 0, image.shape[0], image.shape[1])

        dlib_shape = dlib.full_object_detection(d, landmarks)

        # Get the face descriptor
        face_descriptor = self.dlib_face_recognizer.compute_face_descriptor(image, dlib_shape)
        face_id = self.get_face_id(face_descriptor, [roi.x_offset, roi.y_offset, roi.height, roi.width],
                                   FaceIdRecogniser.current_time_milliseconds())

        if face_id is None:
            face_id = ""

        return face_id
