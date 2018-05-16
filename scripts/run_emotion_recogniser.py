#!/usr/bin/python
from cv_bridge import CvBridge

import recognisers as rp
import rospy
import tensorflow as tf
from recognisers.emotion import EmotionRecogniser
from ros_people_model.srv import Emotion
from ros_people_model.srv import EmotionResponse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    face_landmarks = rp.math.geometry_msgs_points_to_face_landmarks(req.landmarks)
    results = recogniser.recognize(image, face_landmarks)
    return EmotionResponse(results)


if __name__ == "__main__":
    rospy.init_node('emotion_recogniser_server')

    bridge = CvBridge()
    recogniser = EmotionRecogniser()
    recogniser.initialise()
    srv = rospy.Service('emotion_recogniser', Emotion, handle_request)

    rospy.spin()
