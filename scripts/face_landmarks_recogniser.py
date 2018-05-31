#!/usr/bin/python
from cv_bridge import CvBridge

import recognisers as rp
import rospy
from recognisers.face_landmarks import FaceLandmarksRecogniser
from ros_people_model.srv import FaceLandmarks
from ros_people_model.srv import FaceLandmarksResponse


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    landmarks = recogniser.recognize(image)
    points = rp.math.face_landmarks_to_geometry_msgs_points(landmarks)
    return FaceLandmarksResponse(points)


if __name__ == "__main__":

    try:
        rospy.init_node('face_landmarks_recogniser_server')

        bridge = CvBridge()
        recogniser = FaceLandmarksRecogniser()
        recogniser.initialise()
        srv = rospy.Service('face_landmarks_recogniser', FaceLandmarks, handle_request)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass