#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ros_peoplemodel.msg import Feature

DRAW_FRAMERATE = 1.0/30.0

IMAGE = None
FACE_CANDIDATES_CNN = []

def debugDraw(self):
    global IMAGE, FACE_CANDIDATES_CNN

    cnn_clr = (0, 0, 255)
    frt_clr = (0, 0, 0)
    txt_clr = (255, 255, 255)
    shp_clr = (255, 255, 255)
    emo_clr = (150, 150, 125)

    frame = IMAGE.copy()
    overlay_cnn = IMAGE.copy()
    overlay = IMAGE.copy()
    highlights = IMAGE.copy()

    for d in FACE_CANDIDATES_CNN:
        cv2.rectangle(overlay_cnn, (d.left(),d.top()), (d.right(),d.bottom()), cnn_clr, -1)

    for i, d in enumerate(candidates):
        cv2.rectangle(overlay, (d.left(),d.top()), (d.right(),d.bottom()), frt_clr, -1)

    alpha = 0.2
    cv2.addWeighted(overlay_cnn, alpha, frame, 1 - alpha, 0, frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, d in enumerate(candidates):
        if len(faces)-1<i:
            continue

        face_id = faces[i].face_id
        if face_id is not None:
            cv2.putText(frame, face_id[:5], (d.left() + 10, d.top() + 10), cv2.FONT_HERSHEY_PLAIN, 0.9,txt_clr)

        shape = faces[i].shape
        for p in shape:
            cv2.circle(frame, (p.x, p.y), 2, shp_clr)

        emotions = faces[i].emotions
        for p, emo in enumerate(emotions):
            cv2.rectangle(frame, (d.left() + (p*20),      d.bottom() + (int(emo*80))),
                                 (d.left() + (p*20) + 20, d.bottom()), emo_clr, -1)

    cv2.imshow("Image",frame)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        return



def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")

def cnnCallback(data):
    global FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN.append(data)


if __name__ == "__main__":
    rospy.init_node('debug_output', anonymous=True)
    bridge = CvBridge()

    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)
    rospy.Subscriber("/people/vis_dlib_cnn", Feature, cnnCallback)

    # Dlib
    dlib_cnn_detector = dlib.cnn_face_detection_model_v1(DLIB_CNN_MODEL_FILE)
    # Launch detectors
    rospy.Timer(rospy.Duration(CNN_FRATE), debugDraw)

    rospy.spin()
