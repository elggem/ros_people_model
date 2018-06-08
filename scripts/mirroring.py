#!/usr/bin/python
import rospy
from blender_api_msgs.msg import EmotionState, SetGesture
from blender_api_msgs.srv import SetParam
from ros_people_model.msg import Faces
from threading import Lock
import time
import random


class Mirroring:
    PERSON_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    ROBOT_EMOTIONS = ['angry', 'disgust', 'fear', 'happy.001', 'sad', 'surprised', 'none']
    WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    DECAY = 0.08
    BLINK_THRESH = 0.35
    BLINK_THRESH_TIME = 3.0

    def __init__(self):
        self.states_lock = Lock()
        self.states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.emotion_pub = rospy.Publisher('/blender_api/set_emotion_value', EmotionState, queue_size=1)
        self.gesture_pub = rospy.Publisher("/blender_api/set_gesture", SetGesture, queue_size=1)
        self.blender_set_param_srv = rospy.ServiceProxy('/blender_api/set_param', SetParam)

        self.biggest_face = None
        self.last_blink_time = time.time()
        self.blink_state = True
        self.blink_animations = ['blink', 'blink-sleepy', 'blink-relaxed']

        self.fps = rospy.get_param('~fps', 20)

        rospy.Subscriber("/faces", Faces, self.faces_perceived_cb)
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.update_robot_emotions_cb)

    def get_emotion_value(self, emotype):
        return self.biggest_face.emotions[Mirroring.PERSON_EMOTIONS.index(emotype)]

    def update_blink_state(self, value):
        try:
            self.blender_set_param_srv("bpy.data.scenes[\"Scene\"].actuators.ACT_blink_randomly.HEAD_PARAM_enabled", str(value))
            rospy.logdebug("blink_cancel: blender_api/set_param service called")
        except rospy.ServiceException, e:
            rospy.logerr("blink_cancel: blender_api/set_param service call failed %s" % e)

    def blink(self):
        msg = SetGesture()
        msg.name = random.choice(self.blink_animations)
        msg.speed = 1.0
        msg.magnitude = 1.0
        msg.repeat = 0

        self.gesture_pub.publish(msg)

    def update_robot_emotions_cb(self, event):
        with self.states_lock:
            rospy.logdebug(' '.join(['%s: %.3f' % (Mirroring.PERSON_EMOTIONS[n], self.states[n]) for n in
                                     xrange(len(Mirroring.PERSON_EMOTIONS))]))

            for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                expression = EmotionState()
                expression.name = Mirroring.ROBOT_EMOTIONS[i]
                expression.magnitude = self.states[i]
                if expression.magnitude > 0.005:
                    self.emotion_pub.publish(expression)

    def faces_perceived_cb(self, faces):
        with self.states_lock:
            if len(faces.faces) > 0:
                self.biggest_face = faces.faces[0]
                for face in faces.faces:
                    if self.biggest_face.position.z < face.position.z:
                        self.biggest_face = face

                if len(self.biggest_face.emotions) > 1:
                    try:
                        for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                            if self.get_emotion_value(emo) > Mirroring.THRESHOLD[i]:
                                self.states[i] = (self.states[i] * (1.0 - Mirroring.WEIGHTS[i])) + (
                                            Mirroring.WEIGHTS[i] * self.get_emotion_value(emo))
                            else:
                                self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)
                    except:
                        pass
                else:
                    for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                        self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)

                eye_states = self.biggest_face.eye_states
                if len(eye_states) > 1:
                    if self.blink_state:
                        self.update_blink_state(False)
                        self.blink_state = False

                    left = eye_states[0]
                    right = eye_states[1]
                    # print("eye state: ", left, right)
                    diff = time.time() - self.last_blink_time

                    if (left < Mirroring.BLINK_THRESH or right < Mirroring.BLINK_THRESH) and diff > Mirroring.BLINK_THRESH_TIME:
                        self.blink()
                        self.last_blink_time = time.time()
            else:
                self.biggest_face = None

                if not self.blink_state:
                    self.update_blink_state(True)
                for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                    self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)


if __name__ == "__main__":

    try:
        rospy.loginfo("init")
        rospy.init_node('Mirroring')
        node = Mirroring()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
