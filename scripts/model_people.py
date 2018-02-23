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

FACES = []

def updateModelDecay(self):
    global ACTIVE_PEOPLE
    to_remove = []
    for identifier, stored in ACTIVE_PEOPLE.iteritems():
        stored['active'] -= 0.01
        if stored['active'] < 0.0:
            to_remove.append(identifier)
    for rem in to_remove:
        del ACTIVE_PEOPLE[rem]


def positionIsClose(p1,p2,close=0.5):
    return np.linalg.norm([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z]) < close

def blendPositions(p1, p2, bld_pos=0.65, bld_z=0.95):
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
        if positionIsClose(featurePosition, face.position, 200.0):
            print "old face"
            face.position = blendPositions(featurePosition, face.position)
            face.crop = feature.crop
            face.shapes = feature.shapes
            face.emotions = feature.emotions
            face.eyes_closed = feature.eyes_closed
            return

    print "new face"
    face = Face()
    face.crop = feature.crop
    face.position = positionOfFeature(feature)
    face.shapes = feature.shapes
    face.emotions = feature.emotions
    face.eyes_closed = feature.eyes_closed
    FACES.append(face)


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

    #rospy.Subscriber("/people/vis_dlib_cnn", Features, featuresPerceived)
    rospy.Subscriber("/vis_dlib_frontal", Features, featuresPerceived)

    #rospy.Timer(rospy.Duration(1.0/16.0), updateModelDecay)
    rospy.Timer(rospy.Duration(1.0/30.0), publishFaces)

    rospy.spin()
