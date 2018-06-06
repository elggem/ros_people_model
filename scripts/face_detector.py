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
import tf
import math


class FaceDetectorNode(object):

    def __init__(self, recogniser):
        self.recogniser = recogniser

        # config
        self.cnn_scale = rospy.get_param('~scale', 0.4)
        self.cnn_padding = rospy.get_param('~padding', 0.1)
        self.cnn_frate = 1.0 / rospy.get_param('~fps', 5.0)

        # camera settings
        self.aspect = 1.0
        self.fovy = 1.0
        self.rs_depth_scale = 0.001

        self.bridge = CvBridge()
        self.lock = Lock()
        self.image = None
        self.depth = None
        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        self.depth_image = None

        # pubs and subs
        self.faces_pub = rospy.Publisher('/vis_dlib_cnn', Features, queue_size=10)
        self.img_sub = rospy.Subscriber(rospy.get_param('~topic_name', '/camera/image_raw'), Image, self.image_cb)
        self.depth_sub = rospy.Subscriber(
            rospy.get_param('~depth_topic_name', '/camera/aligned_depth_to_color/image_raw'), Image, self.depth_cb)
        rospy.Timer(rospy.Duration(self.cnn_frate), self.face_detect_cb)

    def image_cb(self, msg):
        if self.lock.acquire(False):
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.lock.release()

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def get_world_coordinates(self, xw, x_size, y_size, xc, yc):
        q = float(y_size) / (2.0 * math.tan(self.fovy / 2.0))

        # normalize face position to -1..1,-1..1, where 0,0 is the center of the camera image
        nfx = -1.0 + 2.0 * float(xc) / float(x_size)
        nfy = 1.0 - 2.0 * float(yc) / float(y_size)

        y = -nfx * float(y_size / 2) * self.aspect * xw / q
        z = nfy * float(y_size / 2) * xw / q

        return xw, y, z

    def send_face_transform(self, x, y, z):
        self.br.sendTransform((x, y, z),
                              tf.transformations.quaternion_from_euler(0, 0, 1),
                              rospy.Time.now(),
                              "closest_face",
                              "camera_depth_frame")

    def face_detect_cb(self, event):
        if self.image is not None and self.lock.acquire(False):
            cnn_results = self.recogniser.detect_faces(self.image, scale=self.cnn_scale)

            features = Features()
            features.features = []
            closest_face = None
            max_distance = float("inf")

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

                if self.depth_image is not None:
                    xw = np.array(self.depth_image[roi.y_offset:roi.y_offset + roi.height,
                                  roi.x_offset:roi.x_offset + roi.width]).mean() * self.rs_depth_scale
                else:
                    xw = 2.0

                xc = roi.x_offset + roi.height / 2.0
                yc = roi.y_offset + roi.width / 2.0
                x, y, z = self.get_world_coordinates(xw, self.image.shape[1], self.image.shape[0], xc, yc)
                distance = math.sqrt(x * x + y * y + z * z)

                if distance < max_distance:
                    closest_face = x, y, z

                features.features.append(feature)
            self.lock.release()
            self.faces_pub.publish(features)

            if closest_face is not None:
                x, y, z = closest_face
                self.send_face_transform(x, y, z)


if __name__ == "__main__":

    try:
        rospy.init_node('face_detector')
        recogniser = FaceRecogniser()
        recogniser.initialise()
        node = FaceDetectorNode(recogniser)

        rospy.spin()
    except rospy.ROSInterruptException:
        pass