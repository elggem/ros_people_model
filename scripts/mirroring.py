#!/usr/bin/python
import rospy
from blender_api_msgs.msg import EmotionState
from ros_peoplemodel.msg import Faces

EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'none']
WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

STATES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

DECAY = 0.08


class Tracking:
    def __init__(self):
        self.expressions_pub = rospy.Publisher('/blender_api/set_emotion_value', EmotionState, queue_size=1)
        self.biggestFace = None

        rospy.Subscriber("/faces", Faces, self.facesPerceived)
        rospy.Timer(rospy.Duration(1.0 / 25.0), self.updateHeadPosition)

    def getEmotionValue(self, emotype):
        return self.biggestFace.emotions[EMOTIONS.index(emotype)]

    def updateHeadPosition(self, evt):
        print
        ' '.join(['%s: %.3f' % (EMOTIONS[n], STATES[n]) for n in xrange(len(EMOTIONS))])

        for i, emo in enumerate(EMOTIONS):
            expression = EmotionState()
            expression.name = EXPRESSIONS[i]
            expression.magnitude = STATES[i]
            if expression.magnitude > 0.005:
                self.expressions_pub.publish(expression)

    def facesPerceived(self, faces):
        if len(faces.faces) > 0:
            self.biggestFace = faces.faces[0]
            for face in faces.faces:
                if self.biggestFace.position.z < face.position.z:
                    self.biggestFace = face

            if len(self.biggestFace.emotions) > 1:
                try:
                    for i, emo in enumerate(EMOTIONS):
                        if self.getEmotionValue(emo) > THRESHOLD[i]:
                            STATES[i] = (STATES[i] * (1.0 - WEIGHTS[i])) + (WEIGHTS[i] * self.getEmotionValue(emo))
                        else:
                            STATES[i] = STATES[i] * (1.0 - DECAY)
                except:
                    pass
            else:
                for i, emo in enumerate(EMOTIONS):
                    STATES[i] = STATES[i] * (1.0 - DECAY)
        else:
            self.biggestFace = None
            for i, emo in enumerate(EMOTIONS):
                STATES[i] = STATES[i] * (1.0 - DECAY)


print("init")
rospy.init_node('FaceTracking')
node = Tracking()
rospy.spin()
