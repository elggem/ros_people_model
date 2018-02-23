#!/usr/bin/python
import rospy
import sys
import dlib
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from ros_peoplemodel.msg import Features
from ros_peoplemodel.msg import Feature
from ros_peoplemodel.msg import Faces
from ros_peoplemodel.msg import Face

DRAW_FRAMERATE = 1.0/30.0

IMAGE = None
FACES = None

EMOTIONS = {
    0 : "anger",
    1 : "disgust",
    2 : "fear",
    3 : "happy",
    4 : "sad",
    5 : "surprise",
    6 : "neutral"
}

def debugDraw(self):
    global IMAGE, FACES

    if IMAGE is None or FACES is None:
        return

    cnn_clr = (0, 0, 255)
    frt_clr = (0, 0, 0)
    txt_clr = (255, 255, 255)
    shp_clr = (255, 255, 255)
    emo_clr = (150, 150, 125)

    frame = IMAGE.copy()
    frame = cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
    #frame = cv2.blur(frame,(10,10))

    for face in FACES:
        size = int(face.position.z*4)
        px = int(face.position.x - (size/2.0))
        py = int(face.position.y - (size/2.0))

        img = bridge.imgmsg_to_cv2(face.crop, "8UC3")
        img = cv2.resize(img,(size,size))
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)

        cut_frame = frame[py:py+size, px:px+size, :]
        #frame[py:py+size, px:px+size, :]  = img[:cut_frame.shape[0],:cut_frame.shape[1],:]

        for p in face.shapes:
            cv2.circle(frame, (int(px+(p.x*1.1)), int(py+(p.y*1.1))), 1, shp_clr)

        emo_dict = {}

        if len(face.emotions)>0:
            for i, emotype in enumerate(EMOTIONS):
                emo_dict[EMOTIONS[emotype]] = face.emotions[i]

            p = 0
            for emotype, emo in sorted(emo_dict.iteritems(), key=lambda (k,v): (v,k)):
                cv2.rectangle(frame, (px,                 py + size - 7*20 + (p*20)),
                                     (px + (int(emo*80)), py + size - 7*20 + (p*20) + 20), txt_clr, -1)
                cv2.putText(frame, emotype, (px, 15+ py + size - 7*20+ (p*20)), cv2.FONT_HERSHEY_DUPLEX, 0.55,cnn_clr)
                p += 1

        for p, eye in enumerate(face.eyes_closed):
            cv2.rectangle(frame, (px + (p*20),      py + (int(eye*80))),
                                 (px + (p*20) + 20, py), shp_clr, -1)



    cv2.imshow("Image",frame)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        return


def imageCallback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")

def facesCallback(data):
    global FACES
    FACES = data.faces

if __name__ == "__main__":
    rospy.init_node('model_debug_output', anonymous=True)
    bridge = CvBridge()

    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, imageCallback)
    rospy.Subscriber("/faces", Faces, facesCallback)

    # Launch drawing timer
    rospy.Timer(rospy.Duration(DRAW_FRAMERATE), debugDraw)

    rospy.spin()
