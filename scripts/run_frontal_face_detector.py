#!/usr/bin/python

from multiprocessing.pool import ThreadPool
from os.path import expanduser
from threading import Lock

import numpy as np
import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from recognisers.face import FaceRecogniser
from ros_people_model.cfg import RosPeopleModelConfig
from ros_people_model.msg import Feature
from ros_people_model.msg import Features
from ros_people_model.srv import FaceLandmarks, Emotion, EyeState, FaceId
from sensor_msgs.msg import RegionOfInterest


class FrontalFaceDetector(object):
    DLIB_CNN_MODEL_FILE = expanduser("~/.dlib/mmod_cnn.dat")
    DLIB_CNN_MODEL_URL = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"

    def __init__(self, recogniser):
        self.recogniser = recogniser
        self.cfg_lock = Lock()
        self.cfg = None
        self.bridge = CvBridge()

        self.dynamic_reconfigure_srv = Server(RosPeopleModelConfig, self.dynamic_reconfigure_callback)
        self.faces_pub = rospy.Publisher('vis_dlib_frontal', Features, queue_size=10)
        self.frontal_scale = rospy.get_param('~scale', 0.4)

        self.srv_pool = ThreadPool(processes=3)
        self.srv_eye_state = rospy.ServiceProxy('eye_state_recogniser', EyeState, persistent=True)
        self.srv_face_id = rospy.ServiceProxy('face_id_recogniser', FaceId, persistent=True)
        self.srv_emotion = rospy.ServiceProxy('emotion_recogniser', Emotion, persistent=True)
        self.srv_landmarks = rospy.ServiceProxy('face_landmarks_recogniser', FaceLandmarks, persistent=True)

        self.sub = rospy.Subscriber(rospy.get_param('~topic_name', '/vis_dlib_cnn'), Features, self.features_callback)

    def dynamic_reconfigure_callback(self, config, level):
        with self.cfg_lock:
            self.cfg = config
            rospy.logdebug("Dynamic reconfigure callback result: {0}".format(config))
            return config

    def features_callback(self, msg):
        features = Features()
        features.features = []

        # goes through list and only saves the one
        for k, feature in enumerate(msg.features):
            image = self.bridge.imgmsg_to_cv2(feature.crop, "8UC3")
            faces = self.recogniser.detect_frontal_faces(image, scale=self.frontal_scale)

            if len(faces) == 1:
                face = faces[0]
                ftr = Feature()

                roi = RegionOfInterest()
                roi.x_offset = max(feature.roi.x_offset + face.left(), 0)
                roi.y_offset = max(feature.roi.y_offset + face.top(), 0)
                roi.height = max(face.bottom() - face.top(), 0)
                roi.width = max(face.right() - face.left(), 0)

                ftr.roi = roi
                image2 = np.array(image[face.top():face.bottom(),
                                  face.left():face.right(), :])
                ftr.crop = self.bridge.cv2_to_imgmsg(image2)

                with self.cfg_lock:
                    if self.cfg is not None:
                        if self.cfg.run_face_landmarks:
                            try:
                                ftr.shapes = self.srv_landmarks(ftr.crop).landmarks

                                if self.cfg.run_face_id:
                                    face_id_result = self.srv_pool.apply_async(self.srv_face_id, (
                                    ftr.crop, ftr.roi, ftr.shapes))

                                if self.cfg.run_face_emotions:
                                    emotion_result = self.srv_pool.apply_async(self.srv_emotion, (ftr.crop, ftr.shapes))

                                if self.cfg.run_eye_state:
                                    eye_state_result = self.srv_pool.apply_async(self.srv_eye_state, (ftr.crop, ftr.shapes))

                                if self.cfg.run_face_id:
                                    ftr.face_id = face_id_result.get().face_id

                                if self.cfg.run_face_emotions:
                                    ftr.emotions = emotion_result.get().emotions

                                if self.cfg.run_eye_state:
                                    ftr.eyes_closed = eye_state_result.get().eyes_closed
                            except Exception as e:
                                rospy.logerr("Exception getting features: {0}".format(e))

                features.features.append(ftr)

        self.faces_pub.publish(features)


if __name__ == "__main__":
    rospy.init_node('vis_dlib_frontal')
    recogniser = FaceRecogniser()
    recogniser.initialize_model()
    node = FrontalFaceDetector(recogniser)
    rospy.spin()
