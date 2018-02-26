#!/usr/bin/python
import sys
import rospy
import numpy as np
import time, random
import math
from collections import deque

from ros_peoplemodel.msg import Features
from ros_peoplemodel.msg import Feature
from ros_peoplemodel.msg import Face
from ros_peoplemodel.msg import Faces
from geometry_msgs.msg import Point

# Todo:
# relative scaling factor from image size...

FACES = []

def updateModelDecay(self):
    global FACES
    to_remove = []
    for i, face in enumerate(FACES):
        face.certainty *= 0.8
        if face.certainty < 0.001:
            to_remove.append(i)
    for rem in to_remove:
        del FACES[rem]


def positionIsClose(p1,p2,close=0.5):
    return np.linalg.norm([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z]) < close

def blendPositions(p1, p2, bld_pos=0.45, bld_z=0.95):
    pt = Point()
    pt.x = (p1.x * (1.0-bld_pos)) + (p2.x * bld_pos)
    pt.y = (p1.y * (1.0-bld_pos)) + (p2.y * bld_pos)
    pt.z = (p1.z * (1.0-bld_z)) + (p2.z * bld_z)
    return pt


def positionOfFeature(feature):
    pt = Point()
    pt.x = feature.roi.x_offset + (feature.roi.width / 2)
    pt.y = feature.roi.y_offset + (feature.roi.height / 2)
    pt.z = feature.roi.width * feature.roi.height * 0.00095
    return pt

def updateModelFromFeature(feature):
    global FACES
    for face in FACES:
        featurePosition = positionOfFeature(feature)
        if positionIsClose(featurePosition, face.position, 200.0) or face.face_id == feature.face_id:
            face.position = blendPositions(featurePosition, face.position)
            face.crop = feature.crop
            face.shapes = feature.shapes
            face.emotions = feature.emotions
            face.eyes_closed = feature.eyes_closed
            face.face_id = feature.face_id
            face.certainty = 1.0
            return

    face = Face()
    face.crop = feature.crop
    face.position = positionOfFeature(feature)
    face.certainty = 1.0

    FACES.append(face)

def updateModelFromCNNFeature(feature):
    # Do not use CNN features for position update or anything, just certainty increase.
    for face in FACES:
        featurePosition = positionOfFeature(feature)
        if positionIsClose(featurePosition, face.position, 200.0) or face.face_id == feature.face_id:
            face.certainty = 1.0
            if face.shapes == []:
                featurePosition.z /= 2.0
                face.position = blendPositions(featurePosition, face.position)
            return

    face = Face()
    face.crop = feature.crop
    face.position = positionOfFeature(feature)
    face.position.z /= 2.0
    face.certainty = 1.0
    FACES.append(face)

def cnnFeaturesPerceived(features):
    for feature in features.features:
        updateModelFromCNNFeature(feature)

def featuresPerceived(features):
    for feature in features.features:
        updateModelFromFeature(feature)


def publishFaces(self):
    global FACES
    fcs = Faces()
    fcs.faces = FACES
    pub.publish(fcs)


if __name__ == "__main__":
    rospy.init_node('model_people', anonymous=True)

    pub = rospy.Publisher('faces', Faces, queue_size=10)

    rospy.Subscriber("/people/vis_dlib_cnn", Features, cnnFeaturesPerceived)
    rospy.Subscriber("/vis_dlib_frontal", Features, featuresPerceived)

    rospy.Timer(rospy.Duration(1.0/30.0), updateModelDecay)
    rospy.Timer(rospy.Duration(1.0/30.0), publishFaces)

    rospy.spin()
