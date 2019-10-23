"""Script to gather training images from web-cam"""
FLAGS = {
    'm': 'gather_married',
    'n': 'gather_non_married',
}


def update_flag(key_press, current_flag, flags):
    if key_press < 0 or chr(key_press) not in flags.keys():
        return current_flag

    key_press = chr(key_press)
    for k in flags.keys():
        if k == key_press and k == current_flag:
            print(f'Stop capturing for {k}')
            return None
        elif k == key_press:
            print(f'Capturing for {k}')
            return k


if __name__ == '__main__':
    import argparse
    import os
    import uuid
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images_dir', default='images',
                    help='Main dir for images to be saved to. '
                         '(they will saved to a subdir defined by FLAGS dict)')
    ap.add_argument('-s', '--snapshot', default=1,
                    help='Should only a snapshot be taken when key pressed (1 if yes)? '
                         'Alternative is a keypress toggles continuous capture mode.')
    args = vars(ap.parse_args())

    is_snapshot = args['snapshot'] == 1
    flag = None

    vidcap = cv2.VideoCapture(0)
    while True:
        grabbed_frame, frame = vidcap.read()
        if not grabbed_frame:
            break

        cv2.imshow('Marriage Cam (ESC to quit)', frame)
        key = cv2.waitKey(10)

        if key == 27:
            break
        else:
            flag = update_flag(key, flag, FLAGS)

        if flag is not None:
            frame_name = 'frame_' + str(uuid.uuid4())
            file_name = os.path.join(args['images_dir'], FLAGS[flag], frame_name + '.jpg')
            cv2.imwrite(file_name, frame)

        if is_snapshot:
            flag = None
