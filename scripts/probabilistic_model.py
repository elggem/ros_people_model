#!/usr/bin/python
import sys
import rospy
from ros_slopp.msg import Face

import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
import matplotlib.pyplot as plt
import time, random
import math
from collections import deque

"""


"""
EMOTIONS = {
    0 : "anger",
    1 : "disgust",
    2 : "fear",
    3 : "happy",
    4 : "sad",
    5 : "surprise",
    6 : "neutral"
}

f = KalmanFilter (dim_x=2, dim_z=1)
f.x = np.array([[2.],    # position
                [0.]])   # velocity
f.F = np.array([[1.,1.],
                [0.,1.]])
f.H = np.array([[1.,0.]])
f.P *= 10.
f.R = 5
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)




def facePerceived(face):
    global i
    f.predict()
    f.update(face.emotions[3])


    print "%.8f, %.8f" % (f.x[0], face.emotions[3])

if __name__ == "__main__":
    rospy.init_node('dlib_node', anonymous=True)

    pub = rospy.Publisher('people', Face, queue_size=10)

    rospy.Subscriber("/faces", Face, facePerceived)


    rospy.spin()
