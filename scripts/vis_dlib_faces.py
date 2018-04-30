#!/usr/bin/python
import cv2
import dlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
from ros_peoplemodel.cfg import RosPeopleModelConfig
from ros_peoplemodel.msg import Feature
from ros_peoplemodel.msg import Features
from ros_peoplemodel.srv import DlibFaceID
from ros_peoplemodel.srv import DlibShapes
from ros_peoplemodel.srv import iCogEmopy
from ros_peoplemodel.srv import iCogEyeState
from sensor_msgs.msg import RegionOfInterest

FACE_CANDIDATES_CNN = None


class FaceAlgorithms:

    def __init__(self):
        self.cfg = None

    def perform_face_detection(self, img, scale=1.0):
        if scale is not 1.0:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # perform CNN detection
        dets = dlib_detector(img, 1)
        # rescale
        return [dlib.rectangle(top=int(d.top() / scale),
                               bottom=int(d.bottom() / scale),
                               left=int(d.left() / scale),
                               right=int(d.right() / scale)) for d in dets]

    def features_callback(self, features):
        global FACE_CANDIDATES_CNN
        FACE_CANDIDATES_CNN = features

    def dynamic_reconfigure_callback(self, config, level):
        self.cfg = config
        rospy.logdebug("Dynamic reconfigure callback result: {0}".format(config))
        return config

    def face_detect_frontal_callback(self, event):
        global FACE_CANDIDATES_CNN

        if FACE_CANDIDATES_CNN is None:
            return

        image = bridge.imgmsg_to_cv2(FACE_CANDIDATES_CNN.image, "8UC3")

        features = Features()
        features.image = FACE_CANDIDATES_CNN.image
        features.features = []

        # goes through list and only saves the one
        for k, feature in enumerate(FACE_CANDIDATES_CNN.features):
            crop = bridge.imgmsg_to_cv2(feature.crop, "8UC3")
            dets = self.perform_face_detection(crop, scale=FRONTAL_SCALE)

            if len(dets) == 1:
                d = dets[0]

                ftr = Feature()

                roi = RegionOfInterest()
                roi.x_offset = max(feature.roi.x_offset + d.left(), 0)
                roi.y_offset = max(feature.roi.y_offset + d.top(), 0)
                roi.height = max(d.bottom() - d.top(), 0)
                roi.width = max(d.right() - d.left(), 0)

                ftr.roi = roi
                ftr.crop = bridge.cv2_to_imgmsg(np.array(image[roi.y_offset:roi.y_offset + roi.height,
                                                         roi.x_offset:roi.x_offset + roi.width, :]))

                if self.cfg is not None:
                    if self.cfg.run_face_landmarks:
                        try:
                            ftr.shapes = srv_dlib_shapes(ftr.crop).shape
                        except Exception:
                            pass

                    if self.cfg.run_face_id:
                        try:
                            ftr.face_id = srv_dlib_faceid(ftr.crop, ftr.roi, ftr.shapes).face_id
                        except Exception:
                            pass

                    if self.cfg.run_face_emotions:
                        try:
                            ftr.emotions = srv_icog_emopy(ftr.crop).emotions
                        except Exception:
                            pass

                    if self.cfg.run_eye_state:
                        try:
                            ftr.eyes_closed = srv_icog_eyestate(ftr.crop).eyes_closed
                        except Exception:
                            pass

                features.features.append(ftr)

        pub.publish(features)


if __name__ == "__main__":
    rospy.init_node('vis_dlib_frontal')
    bridge = CvBridge()

    node = FaceAlgorithms()

    FRONTAL_SCALE = rospy.get_param('~scale', 0.4)
    FRONTAL_FRATE = 1.0 / rospy.get_param('~fps', 5.0)

    # Publishers
    pub = rospy.Publisher('vis_dlib_frontal', Features, queue_size=10)
    # Subscribers
    rospy.Subscriber(rospy.get_param('~topic_name', '/vis_dlib_cnn'), Features, node.features_callback)

    # Dlib
    dlib_detector = dlib.get_frontal_face_detector()

    # Attribute Services
    srv_dlib_shapes = rospy.ServiceProxy('vis_srv_dlib_shapes', DlibShapes, persistent=True)
    srv_dlib_faceid = rospy.ServiceProxy('vis_srv_dlib_id', DlibFaceID, persistent=False)
    srv_icog_emopy = rospy.ServiceProxy('vis_srv_icog_emopy', iCogEmopy, persistent=True)
    srv_icog_eyestate = rospy.ServiceProxy('vis_srv_icog_eyestate', iCogEyeState, persistent=True)

    # Launch detectors
    rospy.Timer(rospy.Duration(FRONTAL_FRATE), node.face_detect_frontal_callback)

    srv = Server(RosPeopleModelConfig, node.dynamic_reconfigure_callback)
    rospy.spin()
