#!/usr/bin/python
import rospy
import sys
import numpy as np
import image_geometry
import tf
from geometry_msgs.msg import Pose
from hr_msgs.msg import pau

"""
this script fuses quaternions from head rotation and eye pitch/yaw into a pose topic
"""

def pauCallback(msg):
    # this doesnt make sense, but it works.
    quaternionHead = [msg.m_headRotation.x,  msg.m_headRotation.y,  msg.m_headRotation.z,  msg.m_headRotation.w]
    pryHead = tf.transformations.euler_from_quaternion(quaternionHead, axes='sxyz')
    rpyHead = [pryHead[1], -pryHead[0], pryHead[2]]
    rpyEyes = [0.0, msg.m_eyeGazeLeftPitch, msg.m_eyeGazeLeftYaw]

    rpyCombined = np.array(rpyHead) + np.array(rpyEyes)
    quaternionCombined = tf.transformations.quaternion_from_euler(rpyCombined[0], -rpyCombined[1], rpyCombined[2])

    pose = Pose()
    pose.orientation.x = quaternionCombined[0]
    pose.orientation.y = quaternionCombined[1]
    pose.orientation.z = quaternionCombined[2]
    pose.orientation.w = quaternionCombined[3]

    pub.publish(pose)

    #print("head yaw 1 %.2f yaw 2 %.2f yaw 3 %.2f" % (rpyHead[0], rpyHead[1], rpyHead[2]))
    #print("eye  yaw 1 %.2f yaw 2 %.2f yaw 3 %.2f" % (rpyEyes[0], rpyEyes[1], rpyEyes[2]))
    #print("cbd  yaw 1 %.2f yaw 2 %.2f yaw 3 %.2f" % (rpyCombined[0], rpyCombined[1], rpyCombined[2]))

if __name__ == "__main__":
    rospy.init_node('pau_pose_estimator', anonymous=True)

    # Publishers
    pub = rospy.Publisher('/pose', Pose, queue_size=10)
    # Subscribers
    rospy.Subscriber("/sophia10/head_pau", pau, pauCallback)

    rospy.spin()
