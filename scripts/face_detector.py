#!/usr/bin/python
from cv_bridge import CvBridge
from threading import Lock

import numpy as np
import rospy
from recognisers.face import FaceRecogniser
from ros_people_model.msg import Feature
from ros_people_model.msg import Features
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest


class FaceDetectorNode(object):

    def __init__(self, recogniser):
        self.recogniser = recogniser

        # config
        self.cnn_scale = rospy.get_param('~scale', 0.4)
        self.cnn_padding = rospy.get_param('~padding', 0.1)
        self.cnn_frate = 1.0 / rospy.get_param('~fps', 5.0)

        self.bridge = CvBridge()
        self.lock = Lock()
        self.image = None

        # pubs and subs
        self.faces_pub = rospy.Publisher('/vis_dlib_cnn', Features, queue_size=10)
        self.img_sub = rospy.Subscriber(rospy.get_param('~topic_name', '/camera/image_raw'), Image, self.image_cb)
        rospy.Timer(rospy.Duration(self.cnn_frate), self.face_detect_cb)

    def image_cb(self, msg):
        if self.lock.acquire(False):
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.lock.release()

    def face_detect_cb(self, event):
        if self.image is not None and self.lock.acquire(False):
            cnn_results = self.recogniser.detect_faces(self.image, scale=self.cnn_scale)

            features = Features()
            features.features = []

            for k, d in enumerate(cnn_results):
                padding = int(self.image.shape[0] * self.cnn_padding)

                feature = Feature()

                roi = RegionOfInterest()
                roi.x_offset = np.maximum(d.left() - padding, 0)
                roi.y_offset = np.maximum(d.top() - padding, 0)
                roi.height = np.minimum(d.bottom() - roi.y_offset + padding, self.image.shape[0])
                roi.width = np.minimum(d.right() - roi.x_offset + padding, self.image.shape[1])

                feature.roi = roi
                feature.crop = self.bridge.cv2_to_imgmsg(np.array(self.image[roi.y_offset:roi.y_offset + roi.height,
                                                                  roi.x_offset:roi.x_offset + roi.width, :]))

                features.features.append(feature)

            self.lock.release()
            self.faces_pub.publish(features)


if __name__ == "__main__":

    try:
        rospy.init_node('face_detector')
        recogniser = FaceRecogniser()
        recogniser.initialise()
        node = FaceDetectorNode(recogniser)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
