#!/usr/bin/python
import cv2
import rospy
from cv_bridge import CvBridge
from ros_people_model.msg import Features
from sensor_msgs.msg import Image

DRAW_FRAMERATE = 1.0 / 30.0

IMAGE = None
FACE_CANDIDATES_CNN = None
FACE_CANDIDATES_FRONTAL = None

EMOTIONS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def debug_draw(self):
    global IMAGE, FACE_CANDIDATES_CNN, FACE_CANDIDATES_FRONTAL

    if IMAGE is None:
        return

    cnn_clr = (0, 0, 255)
    frt_clr = (0, 0, 0)
    txt_clr = (255, 255, 255)
    shp_clr = (255, 255, 255)
    emo_clr = (150, 150, 125)

    frame = IMAGE.copy()
    # frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)

    overlay_cnn = IMAGE.copy()
    overlay = IMAGE.copy()
    highlights = IMAGE.copy()

    if FACE_CANDIDATES_CNN is not None:
        for ftr in FACE_CANDIDATES_CNN.features:
            d = ftr.roi
            cv2.rectangle(overlay_cnn, (d.x_offset, d.y_offset), (d.x_offset + d.width, d.y_offset + d.height), cnn_clr,
                          -1)

    if FACE_CANDIDATES_FRONTAL is not None:
        for ftr in FACE_CANDIDATES_FRONTAL.features:
            d = ftr.roi
            cv2.rectangle(overlay, (d.x_offset, d.y_offset), (d.x_offset + d.width, d.y_offset + d.height), frt_clr, -1)

    alpha = 0.2
    cv2.addWeighted(overlay_cnn, alpha, frame, 1 - alpha, 0, frame)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if FACE_CANDIDATES_FRONTAL is not None:
        for ftr in FACE_CANDIDATES_FRONTAL.features:
            d = ftr.roi

            if ftr.face_id is not None:
                cv2.putText(frame, ftr.face_id[:15], (d.x_offset + 10, d.y_offset - 5), cv2.FONT_HERSHEY_PLAIN, 0.9,
                            txt_clr)

            for p in ftr.shapes:
                cv2.circle(frame, (int(d.x_offset - (p.x * 0.2)), int(d.y_offset + (p.y * 0.2))), 1, shp_clr)

            emo_dict = {}

            if len(ftr.emotions) > 0:
                for i, emotype in enumerate(EMOTIONS):
                    emo_dict[EMOTIONS[emotype]] = ftr.emotions[i]

                p = 0
                for emotype, emo in sorted(emo_dict.iteritems(), key=lambda (k, v): (v, k)):
                    cv2.rectangle(frame, (d.x_offset, d.y_offset + d.height - 7 * 20 + (p * 20)),
                                  (d.x_offset + (int(emo * 80)), d.y_offset + d.height - 7 * 20 + (p * 20) + 20),
                                  txt_clr, -1)
                    cv2.putText(frame, emotype, (d.x_offset, 15 + d.y_offset + d.height - 7 * 20 + (p * 20)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, cnn_clr)
                    p += 1

            for p, eye in enumerate(ftr.eyes_closed):
                cv2.rectangle(frame, (d.x_offset + (p * 20), d.y_offset + (int(eye * 80))),
                              (d.x_offset + (p * 20) + 20, d.y_offset), shp_clr, -1)

    cv2.imshow("Image", frame)
    if (cv2.waitKey(10) & 0xFF == ord('q')):
        return


def image_callback(data):
    global IMAGE
    IMAGE = bridge.imgmsg_to_cv2(data, "bgr8")


def cnn_callback(data):
    global FACE_CANDIDATES_CNN
    FACE_CANDIDATES_CNN = data


def frontal_callback(data):
    global FACE_CANDIDATES_FRONTAL
    FACE_CANDIDATES_FRONTAL = data


if __name__ == "__main__":
    rospy.init_node('debug_output', anonymous=True)
    bridge = CvBridge()

    # Subscribers
    rospy.Subscriber("/camera/image_raw", Image, image_callback)
    rospy.Subscriber("/vis_dlib_cnn", Features, cnn_callback)
    rospy.Subscriber("/vis_dlib_frontal", Features, frontal_callback)

    # Launch drawing timer
    rospy.Timer(rospy.Duration(DRAW_FRAMERATE), debug_draw)

    rospy.spin()
