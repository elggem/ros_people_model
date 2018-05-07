#!/usr/bin/python
import numpy as np
import rospy
from geometry_msgs.msg import Point
from ros_people_model.msg import Face
from ros_people_model.msg import Faces
from ros_people_model.msg import Features

# Todo:
# relative scaling factor from image size...

FACES = []


def update_model_decay(event):
    global FACES
    to_remove = []
    for i, face in enumerate(FACES):
        face.certainty *= 0.8
        if face.certainty < 0.001:
            to_remove.append(i)
    for rem in to_remove:
        del FACES[rem]


def position_is_close(p1, p2, close=0.5):
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]) < close


def blend_positions(p1, p2, bld_pos=0.35, bld_z=0.95):
    pt = Point()
    pt.x = (p1.x * (1.0 - bld_pos)) + (p2.x * bld_pos)
    pt.y = (p1.y * (1.0 - bld_pos)) + (p2.y * bld_pos)
    pt.z = (p1.z * (1.0 - bld_z)) + (p2.z * bld_z)
    return pt


def position_of_feature(feature):
    pt = Point()
    pt.x = feature.roi.x_offset + (feature.roi.width / 2)
    pt.y = feature.roi.y_offset + (feature.roi.height / 2)
    pt.z = feature.roi.width * feature.roi.height * 0.00095
    return pt


def update_model_from_feature(feature):
    global FACES
    for face in FACES:
        feature_position = position_of_feature(feature)
        if position_is_close(feature_position, face.position, 200.0) or face.face_id == feature.face_id:
            face.position = blend_positions(feature_position, face.position)
            face.crop = feature.crop
            face.shapes = feature.shapes
            face.emotions = feature.emotions
            face.eyes_closed = feature.eyes_closed
            face.face_id = feature.face_id
            face.certainty = 1.0
            return

    face = Face()
    face.crop = feature.crop
    face.position = position_of_feature(feature)
    face.certainty = 1.0

    FACES.append(face)


def update_model_from_cnn_feature(feature):
    # Do not use CNN features for position update or anything, just certainty increase.
    for face in FACES:
        feature_position = position_of_feature(feature)
        if position_is_close(feature_position, face.position, 200.0) or face.face_id == feature.face_id:
            face.certainty = 1.0
            if not face.shapes:
                feature_position.z /= 2.0
                face.position = blend_positions(feature_position, face.position)
            return

    face = Face()
    face.crop = feature.crop
    face.position = position_of_feature(feature)
    face.position.z /= 2.0
    face.certainty = 1.0
    FACES.append(face)


def cnn_features_perceived(features):
    for feature in features.features:
        update_model_from_cnn_feature(feature)


def features_perceived(features):
    for feature in features.features:
        update_model_from_feature(feature)


def publish_faces(self):
    global FACES
    fcs = Faces()
    fcs.faces = FACES
    pub.publish(fcs)


if __name__ == "__main__":
    rospy.init_node('model_people', anonymous=True)

    pub = rospy.Publisher('faces', Faces, queue_size=10)

    rospy.Subscriber("/vis_dlib_cnn", Features, cnn_features_perceived)
    rospy.Subscriber("/vis_dlib_frontal", Features, features_perceived)

    rospy.Timer(rospy.Duration(1.0 / 30.0), update_model_decay)
    rospy.Timer(rospy.Duration(1.0 / 30.0), publish_faces)

    rospy.spin()
