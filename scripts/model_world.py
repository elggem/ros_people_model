"""
http://docs.ros.org/api/image_geometry/html/python/

use projection to map pixel lvl results into coordinate system and align with pose.


"""

#!/usr/bin/python
import rospy
import sys
import numpy as np
import image_geometry

from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo

CAMERA_MODEL = image_geometry.PinholeCameraModel()
CAMERA_FOCAL_X = 0.0
CAMERA_FOCAL_Y = 0.0


def cameraInfoCallback(msg):
    global CAMERA_MODEL, CAMERA_FOCAL_X, CAMERA_FOCAL_Y
    CAMERA_MODEL.fromCameraInfo(msg)
    CAMERA_FOCAL_X = CAMERA_MODEL.fx()
    CAMERA_FOCAL_Y = CAMERA_MODEL.fy()

if __name__ == "__main__":
    rospy.init_node('pau_pose_estimator', anonymous=True)

    # Publishers
    pub = rospy.Publisher('/pose', Pose, queue_size=10)
    # Subscribers
    rospy.Subscriber("/camera/camera_info", CameraInfo, cameraInfoCallback)

    # Launch drawing timer
    rospy.Timer(rospy.Duration(1.0/10.0), outputValues)

    rospy.spin()
