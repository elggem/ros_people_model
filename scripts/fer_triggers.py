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
    WEIGHTS =    [0.5, 0.5, 0.5, 0.95, 0.5, 0.95, 0.0]
    THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    DECAY = 0.08
    BLINK_THRESH = 0.35
    BLINK_THRESH_TIME = 3.0

    # TRIGGER THRESHOLDS. EVERYTHING ABOVE TRIGGERS
    TRIGGER = [0.9, 0.9, 0.9, 0.5, 0.9, 0.5, 0.9]
    TRIGGER_THRESH_TIME = 2.0

    def __init__(self):
        self.states_lock = Lock()
        self.states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.last_blink_time = time.time()
        self.blink_state = True
        self.blink_animations = ['blink', 'blink-sleepy', 'blink-relaxed']

        self.last_emotion_trigger_time = time.time()

        try:
          robot_name = rospy.get_param('/robot_name')
        except: 
          robot_name = "DEBUG"

        self.chat_trigger_pub = rospy.Publisher('/'+robot_name+'/chatbot_speech', ChatMessage, queue_size=1)
        print('/'+robot_name+'/chatbot_speech')        

        rospy.Subscriber("/faces", Faces, self.faces_perceived_cb)

        rospy.spin()

    #  TRIGGERS
    def triggered_expression(self, expression_name):
        msg = ChatMessage()
        msg.lang = "en-US"
        msg.utterance = "Event.emotion."+expression_name
        msg.source = "ros_people_model"
        print("publishing " + msg.utterance)
        self.chat_trigger_pub.publish(msg)

    def triggered_blink(self):
        pass
        # print("blink detected")


    def get_biggest_face(self, faces):
        biggest_face = faces.faces[0]
        for face in faces.faces:
            if biggest_face.position.z < face.position.z:
                biggest_face = face
        return biggest_face

    def update_emotion_state(self, face):
        # print(face)
        if len(face.emotions) > 1:
            try:
                for i, emo in enumerate(FERTriggers.PERSON_EMOTIONS):
                    if self.get_emotion_value(face, emo) > FERTriggers.THRESHOLD[i]:
                        self.states[i] = (self.states[i] * (1.0 - FERTriggers.WEIGHTS[i])) + (FERTriggers.WEIGHTS[i] * self.get_emotion_value(face, emo))
                    else:
                        self.states[i] = self.states[i] * (1.0 - FERTriggers.DECAY)
            except:
                pass
        else:
            for i, emo in enumerate(FERTriggers.PERSON_EMOTIONS):
                self.states[i] = self.states[i] * (1.0 - FERTriggers.DECAY)

    def print_emotion_state(self):
        for i, emo in enumerate(FERTriggers.PERSON_EMOTIONS):  
            if emo is "happy":    
               print("Emotion: {} State: {}".format(emo, self.states[i]))


    def get_emotion_value(self, face, emotype):
        return face.emotions[FERTriggers.PERSON_EMOTIONS.index(emotype)] * FERTriggers.MULTIPLIER[FERTriggers.PERSON_EMOTIONS.index(emotype)]

    def evaluate_triggers(self):
        for i, emo in enumerate(FERTriggers.PERSON_EMOTIONS):
            if self.states[i] > FERTriggers.TRIGGER[i]:
                diff = time.time() - self.last_emotion_trigger_time
                if diff > FERTriggers.TRIGGER_THRESH_TIME:
                    self.last_emotion_trigger_time = time.time()
                    self.triggered_expression(emo)

    def faces_perceived_cb(self, faces):
        with self.states_lock:
            #self.print_emotion_state()
            self.evaluate_triggers()

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

                    if (left < FERTriggers.BLINK_THRESH or right < FERTriggers.BLINK_THRESH) and diff > FERTriggers.BLINK_THRESH_TIME:
                        self.triggered_blink()
                        self.last_blink_time = time.time()
            else:
                for i, emo in enumerate(FERTriggers.PERSON_EMOTIONS):
                    self.states[i] = self.states[i] * (1.0 - FERTriggers.DECAY)


if __name__ == "__main__":
    try:
        rospy.loginfo("init")
        rospy.init_node('FERTriggers')
        node = FERTriggers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
