
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
from blender_api_msgs.msg import Target, EmotionState, SetGesture
from hr_msgs.msg import TTS
from hr_msgs.msg import pau

"""

Simple demo script that takes the output of ros_peoplemodel to enable face_tracking on Sophia

Used also for Sonar 2017


UNFINISHED
"""

class Tracking:
    def __init__(self):
        self.robot_name = "sophia10"
        self.head_focus_pub = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_focus_pub = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.setpau_pub = rospy.Publisher('/blender_api/set_pau', pau, queue_size=1)
        self.tts_pub = rospy.Publisher('/{}/tts'.format(self.robot_name), TTS, queue_size=1)  # for debug messages
        #self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        #640 360

        self.currentTargetGaze = Target()
        self.currentTargetHead = Target()
        self.biggestFace = None

        self.refactory = 0
        self.center = [710, 360]

        rospy.Subscriber("/faces", Faces, self.facesPerceived)
        rospy.Timer(rospy.Duration(1.0/1.0), self.updateHeadPosition)



    def Say(self,text):
        # publish TTS message
        msg = TTS()
        msg.text = text
        msg.lang = 'en-US'
        self.tts_pub.publish(msg)

    def SetGazeFocus(self,pos,speed):
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = speed
        self.gaze_focus_pub.publish(msg)


    def SetHeadFocus(self,pos,speed):
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = speed
        self.head_focus_pub.publish(msg)

    def updateHeadPosition(self, evt):
        #if self.refactory >= 0:
        #    self.refactory -= 1
        #    print("refactrt: %.2f" % (self.refactory))
        #    return

        if self.biggestFace is None:
            return

        facePosition = self.biggestFace.position

        distanceY = (self.center[0] - facePosition.x) / 1280.0
        distanceZ = (self.center[1] - facePosition.y) / 720.0

        print("Y: %.2f, Z: %.2f" % (distanceY, distanceZ))

        #self.refactory = 5 * np.abs(distanceY + distanceZ)

        #if self.refactory > 10.0:
        self.currentTargetGaze.y += distanceY * 0.2
        self.currentTargetGaze.z += distanceZ * 0.2
        self.currentTargetGaze.speed = 1.0
        self.gaze_focus_pub.publish(self.currentTargetGaze)
        #else:
        #self.currentTargetGaze.y += distanceY * 0.06
        #self.currentTargetGaze.z += distanceZ * 0.06
        self.currentTargetGaze.speed = 0.5
        self.head_focus_pub.publish(self.currentTargetGaze)

        if len(self.biggestFace.emotions)>1:
            happy = self.biggestFace.emotions[3]
            msg = pau()
            msg.m_coeffs = [happy, happy]
            msg.m_shapekeys = ['lips-smile.L', 'lips-smile.R']
            self.setpau_pub.publish(msg)
            print("happy: %.2f" % (happy))





    def facesPerceived(self,faces):

        if len(faces.faces) > 0:
            self.biggestFace = faces.faces[0]
            for face in faces.faces:
                if self.biggestFace.position.z < face.position.z:
                    self.biggestFace = face
        else:
            self.biggestFace = None



if __name__ == "__main__":
    rospy.init_node('FaceTracking')
    node = Tracking()
    rospy.spin()
