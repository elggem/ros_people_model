#!/usr/bin/python
import rospy
from blender_api_msgs.msg import EmotionState
from ros_people_model.msg import Faces
from threading import Lock


class Mirroring:
    PERSON_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    ROBOT_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'none']
    WEIGHTS = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    THRESHOLD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    DECAY = 0.08

    def __init__(self):
        self.states_lock = Lock()
        self.states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.emotion_pub = rospy.Publisher('/blender_api/set_emotion_value', EmotionState, queue_size=1)
        self.biggest_face = None

        self.fps = rospy.get_param('~fps', 20)

        rospy.Subscriber("/faces", Faces, self.faces_perceived_cb)
        rospy.Timer(rospy.Duration(1.0 / self.fps), self.update_robot_emotions_cb)

    def get_emotion_value(self, emotype):
        return self.biggest_face.emotions[Mirroring.PERSON_EMOTIONS.index(emotype)]

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
            else:
                self.biggest_face = None
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
