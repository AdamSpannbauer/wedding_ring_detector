import os
import uuid
import cv2

images_dir = 'images'
vidcap = cv2.VideoCapture(0)
while True:
    grabbed_frame, frame = vidcap.read()
    if not grabbed_frame:
        break

    frame_name = 'frame_' + str(uuid.uuid4())

    cv2.imshow('Marriage Cam', frame)
    key = cv2.waitKey(10)

    if key == 27 or key == ord('q'):
        break
    elif key == ord('m'):
        file_name = os.path.join(images_dir, 'gather_married', frame_name + '.jpg')
        cv2.imwrite(file_name, frame)
    elif key == ord('n'):
        file_name = os.path.join(images_dir, 'gather_non_married', frame_name + '.jpg')
        cv2.imwrite(file_name, frame)
