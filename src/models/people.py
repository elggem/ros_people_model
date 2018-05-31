from threading import Lock

import models
import rospy
from ros_people_model.msg import Face
from ros_people_model.msg import Faces
from ros_people_model.msg import Features


class PeopleModel(object):
    """
     Todo: relative scaling factor from image size...
    """

    def __init__(self):
        self.faces_lock = Lock()
        self.faces = []

        self.fps = rospy.get_param('~fps', 20)
        self.faces_pub = rospy.Publisher('faces', Faces, queue_size=10)

        rospy.Subscriber("/vis_dlib_cnn", Features, self.cnn_features_perceived_cb)
        rospy.Subscriber("/vis_dlib_frontal", Features, self.features_perceived_cb)

        duration = 1.0 / self.fps
        rospy.Timer(rospy.Duration(duration), self.update_model_decay_cb)
        rospy.Timer(rospy.Duration(duration), self.publish_faces_cb)

    def update_model_from_feature(self, feature):
        for face in self.faces:
            feature_position = models.math.position_of_feature(feature)
            if models.math.position_is_close(feature_position, face.position, 200.0) or face.face_id == feature.face_id:
                face.position = models.math.blend_positions(feature_position, face.position)
                face.crop = feature.crop
                face.shapes = feature.shapes
                face.emotions = feature.emotions
                face.eye_states = feature.eye_states
                face.face_id = feature.face_id
                face.certainty = 1.0
                return

        face = Face()
        face.crop = feature.crop
        face.position = models.math.position_of_feature(feature)
        face.certainty = 1.0

        self.faces.append(face)

    def update_model_from_cnn_feature(self, feature):
        # Do not use CNN features for position update or anything, just certainty increase.
        for face in self.faces:
            feature_position = models.math.position_of_feature(feature)
            if models.math.position_is_close(feature_position, face.position, 200.0) or face.face_id == feature.face_id:
                face.certainty = 1.0
                if not face.shapes:
                    feature_position.z /= 2.0
                    face.position = models.math.blend_positions(feature_position, face.position)
                return

        face = Face()
        face.crop = feature.crop
        face.position = models.math.position_of_feature(feature)
        face.position.z /= 2.0
        face.certainty = 1.0
        self.faces.append(face)

    def update_model_decay_cb(self, event):
        with self.faces_lock:
            to_remove = []
            for i, face in enumerate(self.faces):
                face.certainty *= 0.8
                if face.certainty < 0.001:
                    to_remove.append(i)

            # we must delete items in reverse order otherwise in subsequent iterations we might try to delete indices
            # that don't exist in the list anymore
            for i in sorted(to_remove, reverse=True):
                del self.faces[i]

    def cnn_features_perceived_cb(self, features):
        with self.faces_lock:
            for feature in features.features:
                self.update_model_from_cnn_feature(feature)

    def features_perceived_cb(self, features):
        with self.faces_lock:
            for feature in features.features:
                self.update_model_from_feature(feature)

    def publish_faces_cb(self):
        with self.faces_lock:
            msg = Faces()
            msg.faces = self.faces
            self.faces_pub.publish(msg)
