#!/usr/bin/python
import rospy
from ros_people_model.msg import Faces
from hr_msgs.msg import ChatMessage
from threading import Lock
import time
import random


class FERTriggers:
    PERSON_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    MULTIPLIER = [1.05, 1.1, 1.1, 1.07, 1.25, 1.8, 1.0]
    WEIGHTS =    [0.5, 0.5, 0.5, 0.05, 0.5, 0.5, 0.0]
    THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    DECAY = 0.08
    BLINK_THRESH = 0.35
    BLINK_THRESH_TIME = 3.0

    # TRIGGER THRESHOLDS. EVERYTHING ABOVE TRIGGERS
    TRIGGER = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    def __init__(self):
        self.states_lock = Lock()
        self.states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.last_blink_time = time.time()
        self.blink_state = True
        self.blink_animations = ['blink', 'blink-sleepy', 'blink-relaxed']

        self.fps = rospy.get_param('~fps', 20)

        robot_name = rospy.get_param('/robot_name')
        self.chat_trigger_pub = rospy.Publisher('/'+robot_name+'/speech', ChatMessage, queue_size=1)

        rospy.Subscriber("/faces", Faces, self.faces_perceived_cb)
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.update_robot_emotions_cb)

    #  TRIGGERS
    def triggered_expression(self, expression_name):
        msg = ChatMessage()
        msg.lang = "en-US"
        utterance = "expression."+expression_name
        print("publishing " + utterance)
        self.chat_trigger_pub(msg)

    def triggered_blink(self):
        print("blink detected")


    def get_biggest_face(faces):
        biggest_face = faces.faces[0]
        for face in faces.faces:
            if biggest_face.position.z < face.position.z:
                biggest_face = face
        return faces

    def update_emotion_state(face):
        if len(face.emotions) > 1:
            try:
                for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                    if self.get_emotion_value(face, emo) > Mirroring.THRESHOLD[i]:
                        self.states[i] = (self.states[i] * (1.0 - Mirroring.WEIGHTS[i])) + (
                                    Mirroring.WEIGHTS[i] * self.get_emotion_value(face, emo))
                    else:
                        self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)
            except:
                pass
        else:
            for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)

    def get_emotion_value(self, face, emotype):
        return face.emotions[Mirroring.PERSON_EMOTIONS.index(emotype)] * Mirroring.MULTIPLIER[Mirroring.PERSON_EMOTIONS.index(emotype)]

    def evaluate_triggers(self, face):
        for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
            if self.get_emotion_value(face, emo) > Mirroring.TRIGGER[i]:
                self.triggered_expression(emo)

    def faces_perceived_cb(self, faces):
        with self.states_lock:
            if len(faces.faces) > 0:
                # Get biggest face
                biggest_face = self.get_biggest_face(faces)

                # Update Emotion states
                self.update_emotion_state(biggest_face)

                # Update eyes
                eye_states = biggest_face.eye_states
                if len(eye_states) > 1:
                    if self.blink_state:
                        self.blink_state = False

                    left = eye_states[0]
                    right = eye_states[1]
                    # print("eye state: ", left, right)
                    diff = time.time() - self.last_blink_time

                    if (left < Mirroring.BLINK_THRESH or right < Mirroring.BLINK_THRESH) and diff > Mirroring.BLINK_THRESH_TIME:
                        self.triggered_blink()
                        self.last_blink_time = time.time()
            else:
                for i, emo in enumerate(Mirroring.PERSON_EMOTIONS):
                    self.states[i] = self.states[i] * (1.0 - Mirroring.DECAY)


if __name__ == "__main__":
    try:
        rospy.loginfo("init")
        rospy.init_node('FERTriggers')
        node = FERTriggers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
