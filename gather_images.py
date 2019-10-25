"""Script to gather training images from web-cam"""
import os
import string
import uuid
import cv2


def update_flag(key_press, current_flag, flags):
    """Handle key press from cv2.waitKey() for capturing frames

    :param key_press: output from `cv2.waitKey()`
    :param current_flag: value of 'flag' holding previous key press
    :param flags: dictionary mapping key presses to class labels
    :return: new flag value
    """
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


def try_int(x):
    """Helper to accept int and str from argparse for cv2.VideoCapture

    :param x: input from arparse
    :return: x converted to int; if Value error x returned unchanged
    """
    try:
        return int(x)
    except ValueError:
        return x


def prompt_labels(n_class=2):
    """Prompt user for class labels and map them to keys for gathering training data

    :param n_class: number of classes to prompt labels for
    :return: tuple of labels and key press their mapped to
    """
    if n_class > 26:
        raise ValueError('Only supports up to 26 classes')

    keys = list(string.ascii_lowercase[:n_class])

    labels = []
    for key in keys:
        label = input(f'Label for key press "{key}": ')
        labels.append(label)
    return labels, keys


def mkdirs(dir_names):
    """Create dirs if they don't exist

    :param dir_names: names of dirs to create; if nested, provide parent in list before child
    :return: None
    """
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


def gather_images(output_dir, labels=None, n_classes=2, video_capture=0, snapshot=True):
    """Capture training data for building a 2 class model

    :param output_dir: main dir for images to be saved to (they will saved to a subdir named by `labels`)
    :param labels: len 2 list of labels for the classes (a will be key for position 0 and b for 1)
    :param n_classes: number of classes
    :param video_capture: value to pass to `cv2.VideoCapture()`
    :param snapshot: Should only a snapshot be taken when key pressed?
                     If False, a keypress toggles continuous capture mode.
    :return: None; images are saved to output_dir
    """
    if labels is None:
        labels, keys = prompt_labels(n_classes)
    else:
        keys = list(string.ascii_lowercase[:len(labels)])

    label_key_dict = {k: v for k, v in zip(keys, labels)}

    # Ensure dirs exist (create them if not)
    output_sub_dirs = [os.path.join(output_dir, l) for l in labels]
    mkdirs([output_dir] + output_sub_dirs)

    vidcap = cv2.VideoCapture(video_capture)
    capture_flag = None
    while True:
        grabbed_frame, frame = vidcap.read()
        if not grabbed_frame:
            break

        cv2.imshow('Gather Training Data (ESC to quit)', frame)
        key = cv2.waitKey(10)

        if key == 27:
            break
        else:
            capture_flag = update_flag(key, capture_flag, label_key_dict)

        if capture_flag is not None:
            frame_name = 'frame_' + str(uuid.uuid4())
            file_name = os.path.join(output_dir, label_key_dict[capture_flag], frame_name + '.jpg')
            cv2.imwrite(file_name, frame)

        if snapshot:
            capture_flag = None

    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type=try_int, default=0,
                    help='Path to input video. '
                         '(can be number to indicate web cam; see cv2.VideoCapture docs)')
    ap.add_argument('-o', '--output', default='images',
                    help='Main dir for images to be saved to. '
                         '(they will saved to a subdir defined by FLAGS dict)')
    ap.add_argument('-n', '--n_classes', type=int, default=2,
                    help='Number of classes to define')
    ap.add_argument('-s', '--snapshot', default=1,
                    help='Should only a snapshot be taken when key pressed (1 if yes)? '
                         'Alternative is a keypress toggles continuous capture mode.')
    args = vars(ap.parse_args())

    is_snapshot = args['snapshot'] == 1
    gather_images(output_dir=args['output'],
                  video_capture=args['input'],
                  n_classes=args['n_classes'],
                  snapshot=is_snapshot)
