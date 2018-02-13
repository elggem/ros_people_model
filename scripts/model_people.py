#!/usr/bin/python
import sys
import rospy
from ros_slopp.msg import Face

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import time, random
import math
from collections import deque


EMOTIONS = {
    0 : "anger",
    1 : "disgust",
    2 : "fear",
    3 : "happy",
    4 : "sad",
    5 : "surprise",
    6 : "neutral"
}

ACTIVE_PEOPLE = {}

def decayPeople(self):
    global ACTIVE_PEOPLE
    to_remove = []
    for identifier, stored in ACTIVE_PEOPLE.iteritems():
        stored['active'] -= 0.01
        if stored['active'] < 0.0:
            to_remove.append(identifier)
    for rem in to_remove:
        del ACTIVE_PEOPLE[rem]

def facePerceived(face):
    global ACTIVE_PEOPLE
    if face.face_id == "None":
        for identifier, stored in ACTIVE_PEOPLE.iteritems():
            distance = np.linalg.norm(np.array(stored['face'].bounding_box) - np.array(face.bounding_box))
            if distance < 400:
                stored['active'] += 0.01
                stored['active'] = np.minimum(1.0, stored['active'])
                return
        return
    else:
        for identifier, stored in ACTIVE_PEOPLE.iteritems():
            if identifier == face.face_id:
                stored['active'] += 0.01
                stored['active'] = np.minimum(1.0, stored['active'])
                return

    ACTIVE_PEOPLE[face.face_id] = {'face':face, 'active':0.1}

def printPerceived(self):
    global ACTIVE_PEOPLE
    for identifier, stored in ACTIVE_PEOPLE.iteritems():
        if stored['active'] > 0.1:
            print("%s %.3f" % (identifier, stored['active']))
            stored['face'].certainty = stored['active']
            pub.publish(stored['face'])
    print("---")

if __name__ == "__main__":
    rospy.init_node('dlib_node', anonymous=True)

    pub = rospy.Publisher('people', Face, queue_size=10)

    rospy.Subscriber("/faces", Face, facePerceived)

    rospy.Timer(rospy.Duration(1.0/16.0), decayPeople)
    rospy.Timer(rospy.Duration(1.0/8.0), printPerceived)

    rospy.spin()
