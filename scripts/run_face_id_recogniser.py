#!/usr/bin/python
import recognisers as rp
import rospy
from cv_bridge import CvBridge
from recognisers.face_id import FaceIdRecogniser
from ros_people_model.srv import FaceId
from ros_people_model.srv import FaceIdResponse


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    landmarks = rp.math.geometry_msgs_points_to_face_landmarks(req.landmarks)
    result = recogniser.recognize(image, req.roi, landmarks)
    return FaceIdResponse(result)


if __name__ == "__main__":
    bridge = CvBridge()
    recogniser = FaceIdRecogniser()
    recogniser.initialize_models()

    rospy.init_node('face_id_recogniser_server')
    srv = rospy.Service('face_id_recogniser', FaceId, handle_request)

    rospy.spin()