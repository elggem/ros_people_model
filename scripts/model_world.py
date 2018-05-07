#!/usr/bin/python

"""
http://docs.ros.org/api/image_geometry/html/python/

use projection to map pixel lvl results into coordinate system and align with pose.


Can optinally receive /pose topic from orb2slam or pauposeestimator

"""

import image_geometry
import rospy
import tf
from geometry_msgs.msg import Pose
from ros_people_model.msg import Features
from sensor_msgs.msg import CameraInfo

CAMERA_MODEL = image_geometry.PinholeCameraModel()
CAMERA_FOCAL_X = 0.0
CAMERA_FOCAL_Y = 0.0

POSE_QUATERNION = [0.0, 0.0, 0.0, 0.0]


# rotate vector v1 by quaternion q1
# https://answers.ros.org/question/196149/how-to-rotate-vector-by-quaternion-in-python/
def qv_mult(q1, v1):
    v1 = tf.transformations.unit_vector(v1)
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2),
        tf.transformations.quaternion_conjugate(q1)
    )[:3]


def project_pixel_and_pose_to_3d_ray(uv):
    global POSE_QUATERNION
    return qv_mult(POSE_QUATERNION, CAMERA_MODEL.projectPixelTo3dRay(uv))


def pose_callback(pose):
    global POSE_QUATERNION
    POSE_QUATERNION = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]


def camera_info_callback(msg):
    global CAMERA_MODEL
    CAMERA_MODEL.fromCameraInfo(msg)


def features_callback(features):
    for roi in features.rois:
        uv = (roi.x_offset + (roi.width / 2), roi.y_offset + (roi.height / 2))
        ray = project_pixel_and_pose_to_3d_ray(uv)
        rospy.logdebug(ray)
    pass


if __name__ == "__main__":
    rospy.init_node('model_world', anonymous=True)

    # Publishers
    # pub = rospy.Publisher('/pose', Pose, queue_size=10)
    # Subscribers
    rospy.Subscriber("/pose", Pose, pose_callback)
    rospy.Subscriber("/camera/camera_info", CameraInfo, camera_info_callback)

    # TODO: replace with people model
    rospy.Subscriber("/people/vis_dlib_cnn", Features, features_callback)

    rospy.spin()
