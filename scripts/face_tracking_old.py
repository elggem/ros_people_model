# !/usr/bin/python
import rospy
from blender_api_msgs.msg import Target
from hr_msgs.msg import TTS
from hr_msgs.msg import pau
from ros_peoplemodel.msg import Faces

"""

Simple demo script that takes the output of ros_peoplemodel to enable face_tracking on Sophia

Used also for Sonar 2017

UNFINISHED
"""


class Tracking:
    def __init__(self):
        self.robot_name = "sophia14"
        self.head_focus_pub = rospy.Publisher('/blender_api/set_face_target', Target, queue_size=1)
        self.gaze_focus_pub = rospy.Publisher('/blender_api/set_gaze_target', Target, queue_size=1)
        self.setpau_pub = rospy.Publisher('/blender_api/set_pau', pau, queue_size=1)
        self.tts_pub = rospy.Publisher('/{}/tts'.format(self.robot_name), TTS, queue_size=1)  # for debug messages
        # self.hand_events_pub = rospy.Publisher('/hand_events', String, queue_size=1)

        # 640 360

        self.current_target_gaze = Target()
        self.current_target_head = Target()
        self.biggest_face = None

        self.refactory = 0
        self.center = [480, 270]

        rospy.Subscriber("/faces", Faces, self.faces_perceived)
        rospy.Timer(rospy.Duration(1.0 / 8.0), self.update_head_position)

    def say(self, text):
        # publish TTS message
        msg = TTS()
        msg.text = text
        msg.lang = 'en-US'
        self.tts_pub.publish(msg)

    def set_gaze_focus(self, pos, speed):
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = speed
        self.gaze_focus_pub.publish(msg)

    def set_head_focus(self, pos, speed):
        msg = Target()
        msg.x = pos.x
        msg.y = pos.y
        msg.z = pos.z
        msg.speed = speed
        self.head_focus_pub.publish(msg)

    def update_head_position(self, evt):
        # if self.refactory >= 0:
        #    self.refactory -= 1
        #    print("refactrt: %.2f" % (self.refactory))
        #    return

        if self.biggest_face is None:
            return

        face_position = self.biggest_face.position

        distance_y = (self.center[0] - face_position.x) / 960.0
        distance_z = (self.center[1] - face_position.y) / 540.0

        print("Y: %.2f, Z: %.2f" % (distance_y, distance_z))

        # self.refactory = 5 * np.abs(distanceY + distanceZ)

        # if self.refactory > 10.0:
        self.current_target_gaze.y += distance_y * 0.2
        self.current_target_gaze.z += distance_z * 0.2
        self.current_target_gaze.speed = 1.0
        self.gaze_focus_pub.publish(self.current_target_gaze)
        # else:
        # self.currentTargetGaze.y += distanceY * 0.06
        # self.currentTargetGaze.z += distanceZ * 0.06
        # self.currentTargetGaze.speed = 0.5
        # self.head_focus_pub.publish(self.currentTargetGaze)

        # if len(self.biggestFace.emotions)>1:
        #    happy = self.biggestFace.emotions[3]
        #    msg = pau()
        #    msg.m_coeffs = [happy, happy]
        #    msg.m_shapekeys = ['lips-smile.L', 'lips-smile.R']
        #    self.setpau_pub.publish(msg)
        #    print("happy: %.2f" % (happy))

    def faces_perceived(self, faces):

        if len(faces.faces) > 0:
            self.biggest_face = faces.faces[0]
            for face in faces.faces:
                if self.biggest_face.position.z < face.position.z:
                    self.biggest_face = face
        else:
            self.biggest_face = None


if __name__ == "__main__":
    rospy.init_node('FaceTracking')
    node = Tracking()
    rospy.spin()
