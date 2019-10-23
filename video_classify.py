"""Apply model to frames of video from web-cam"""
import pickle
import cv2
import imutils.text
from extract_features import FeatureExtractor


class MarriagePredictor:
    """Wrapper class to hold binary married/not married hand classifier

    :param model_path: File path to pickled marriage/not married sklearn model
    :param threshold: Cutoff to be considered not married (if married <= threshold then label is not married)
    :ivar threshold: see param threshold
    :ivar labels: Model class labels for [0, 1] classes
    """
    def __init__(self, model_path='model.pickle', threshold=0.5):
        self.threshold = threshold
        self.labels = ['Married', 'Not Married']
        self._feature_extractor = FeatureExtractor()

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, image):
        features = self._feature_extractor.extract_features_cv2(image)
        probabilities = self.model.predict_proba(features)[0]

        label_ind = 0
        if probabilities[0] <= self.threshold:
            label_ind = 1

        return self.labels[label_ind], probabilities


def put_alpha_centered_text(img, text, alpha=0.9,
                            font_face=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=1.5, color=(0, 255, 0), thickness=2):
    """Put transparent text on an image

    :param img: Image to put text on
    :param text: see `cv2.putText`
    :param alpha: Level of transparency in [0-1]; 1 is opaque
    :param font_face: see `cv2.putText`
    :param font_scale: see `cv2.putText`
    :param color: see `cv2.putText`
    :param thickness: see `cv2.putText`
    :return: None; img is modified in place
    """
    beta = 1 - alpha
    beta_layer = img.copy()
    alpha_layer = img.copy()

    imutils.text.put_centered_text(alpha_layer,
                                   text=text,
                                   font_face=font_face,
                                   font_scale=font_scale,
                                   color=color,
                                   thickness=thickness)

    cv2.addWeighted(alpha_layer, alpha, beta_layer, beta, gamma=0, dst=img)


def text_bar_chart(labs, vals, length=10, display_val=False, highlight_max=False,
                   bar_char='-', highlight_char='='):
    """Display ascii barchart

    :param labs: Text labels for values of bars.
    :param vals: Values to be plotted; assumed to be [0-1].
    :param length: Max bar length in characters
    :param display_val: Should value be displayed as text beside bar?
    :param highlight_max: Should should the bar with the max `val` be
                          highlighted in a different color?
    :param bar_char: Character to be used to build bar
    :param highlight_char: Character to be used to build bar of max len bar;
                           only used if highlight_max is True.
    :return: None; `img` is modified in place
    """
    bars = []
    max_val = max(vals)
    for l, v in zip(labs, vals):
        char = bar_char
        if highlight_max and v == max_val:
            char = highlight_char

        label = '  '
        if display_val:
            label += f'({v:.2f})'
        label += f' {l}'

        bar = char * int(round(length * v))
        bars.append(bar + label)

    bar_text_lines = '\n'.join(bars)
    return bar_text_lines


if __name__ == '__main__':
    marriage_predictor = MarriagePredictor(threshold=0.5)

    vidcap = cv2.VideoCapture(0)
    writer = None
    while True:
        grabbed_frame, frame = vidcap.read()
        if not grabbed_frame:
            break

        lab, probs = marriage_predictor.predict(frame)

        put_alpha_centered_text(frame,
                                text=lab,
                                alpha=0.5,
                                font_scale=2,
                                thickness=3)

        bar_text = text_bar_chart(marriage_predictor.labels, probs, highlight_max=True)
        imutils.text.put_text(frame,
                              text=bar_text,
                              org=(10, 25),
                              font_face=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.8,
                              color=(0, 0, 255),
                              thickness=2)

        frame = imutils.resize(frame, width=750)
        if writer is None:
            h, w = frame.shape[:2]

            # setup video writer
            writer = cv2.VideoWriter('video_classifier_output.avi',
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     15, (w, h), True)

        writer.write(frame)

        cv2.imshow('Marriage Cam (ESC to quit)', frame)
        key = cv2.waitKey(10)

        if key == 27:
            break

    writer.release()
