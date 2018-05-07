#!/usr/bin/python
import rospy
from blender_api_msgs.msg import EmotionState
from ros_people_model.msg import Faces

EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'none']
WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
STATES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DECAY = 0.08


class Tracking:
    def __init__(self):
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_value', EmotionState, queue_size=1)
        self.biggest_face = None

        rospy.Subscriber("/faces", Faces, self.faces_perceived)
        rospy.Timer(rospy.Duration(1.0 / 25.0), self.update_head_position)

    def get_emotion_value(self, emotype):
        return self.biggest_face.emotions[EMOTIONS.index(emotype)]

    def update_head_position(self, event):
        rospy.logdebug(' '.join(['%s: %.3f' % (EMOTIONS[n], STATES[n]) for n in xrange(len(EMOTIONS))]))

        for i, emo in enumerate(EMOTIONS):
            expression = EmotionState()
            expression.name = EXPRESSIONS[i]
            expression.magnitude = STATES[i]
            if expression.magnitude > 0.005:
                self.expressions_pub.publish(expression)

    def faces_perceived(self, faces):
        if len(faces.faces) > 0:
            self.biggest_face = faces.faces[0]
            for face in faces.faces:
                if self.biggest_face.position.z < face.position.z:
                    self.biggest_face = face

            if len(self.biggest_face.emotions) > 1:
                try:
                    for i, emo in enumerate(EMOTIONS):
                        if self.get_emotion_value(emo) > THRESHOLD[i]:
                            STATES[i] = (STATES[i] * (1.0 - WEIGHTS[i])) + (WEIGHTS[i] * self.get_emotion_value(emo))
                        else:
                            STATES[i] = STATES[i] * (1.0 - DECAY)
                except:
                    pass
            else:
                for i, emo in enumerate(EMOTIONS):
                    STATES[i] = STATES[i] * (1.0 - DECAY)
        else:
            self.biggest_face = None
            for i, emo in enumerate(EMOTIONS):
                STATES[i] = STATES[i] * (1.0 - DECAY)


rospy.loginfo("init")
rospy.init_node('FaceTracking')
node = Tracking()
rospy.spin()
