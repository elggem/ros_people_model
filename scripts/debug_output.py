#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ros_peoplemodel.msg import Features

DRAW_FRAMERATE = 1.0/30.0

IMAGE = None
FACE_CANDIDATES_CNN = None
FACE_CANDIDATES_FRONTAL = None

def debugDraw(self):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL

    if IMAGE is None:
        return

    cnn_clr = (0, 0, 255)
    frt_clr = (0, 0, 0)
    txt_clr = (255, 255, 255)
    shp_clr = (255, 255, 255)
    emo_clr = (150, 150, 125)

    frame = IMAGE.copy()
    overlay_cnn = IMAGE.copy()
    overlay = IMAGE.copy()
    highlights = IMAGE.copy()

    if FACE_CANDIDATES_CNN is not None:
        for ftr in FACE_CANDIDATES_CNN.features:
            d = ftr.roi
            cv2.rectangle(overlay_cnn, (d.x_offset,d.y_offset), (d.x_offset+d.width,d.y_offset+d.height), cnn_clr, -1)

    if FACE_CANDIDATES_FRONTAL is not None:
        for ftr in FACE_CANDIDATES_FRONTAL.features:
            d = ftr.roi
            cv2.rectangle(overlay, (d.x_offset,d.y_offset), (d.x_offset+d.width,d.y_offset+d.height), frt_clr, -1)

    alpha = 0.2
    cv2.addWeighted(overlay_cnn, alpha, frame, 1 - alpha, 0, frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if FACE_CANDIDATES_FRONTAL is not None:
        for ftr in FACE_CANDIDATES_FRONTAL.features:
            d = ftr.roi

            if ftr.face_id is not None:
                cv2.putText(frame, ftr.face_id[:5], (d.x_offset + 10, d.y_offset + 10), cv2.FONT_HERSHEY_PLAIN, 0.9,txt_clr)

            for p in ftr.shapes:
                cv2.circle(frame, (int(d.x_offset+(p.x*0.2)), int(d.y_offset+(p.y*0.2))), 1, shp_clr)

            for p, emo in enumerate(ftr.emotions):
                cv2.rectangle(frame, (d.x_offset + (p*20),      d.y_offset+d.height + (int(emo*80))),
                                     (d.x_offset + (p*20) + 20, d.y_offset+d.height), emo_clr, -1)

    cv2.imshow("Image",frame)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        return


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")

def cnnCallback(data):
    global FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = data

def frontalCallback(data):
    global FACE_CANDIDATES_FRONTAL
    FACE_CANDIDATES_FRONTAL = data


if __name__ == "__main__":
    rospy.init_node('debug_output', anonymous=True)
    bridge = CvBridge()

    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)
    rospy.Subscriber("/people/vis_dlib_cnn", Features, cnnCallback)
    rospy.Subscriber("/vis_dlib_frontal", Features, frontalCallback)

    # Launch drawing timer
    rospy.Timer(rospy.Duration(DRAW_FRAMERATE), debugDraw)

    rospy.spin()
