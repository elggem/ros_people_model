
def debugDraw(candidates, faces):
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
