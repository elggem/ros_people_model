#!/usr/bin/python
import recognisers as rp
import rospy
from cv_bridge import CvBridge
from recognisers.eye_state import EyeStateRecogniser
from ros_people_model.srv import EyeState
from ros_people_model.srv import EyeStateResponse

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    face_landmarks = rp.math.geometry_msgs_points_to_face_landmarks(req.landmarks)
    result = recogniser.recognize(image, face_landmarks)
    return EyeStateResponse(result)


if __name__ == "__main__":
    bridge = CvBridge()
    recogniser = EyeStateRecogniser()
    recogniser.initialize_models()

    rospy.init_node('eye_state_recogniser_server')
    srv = rospy.Service('eye_state_recogniser', EyeState, handle_request)

    rospy.spin()