import pickle
import cv2
import imutils.text
from extract_features import FeatureExtractor


class MarriagePredictor:
    def __init__(self, model_path='model.pickle', threshold=0.5):
        self.labels = ['Married', 'Not Married']
        self.feature_extractor = FeatureExtractor()
        self.threshold = threshold
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, image):
        features = self.feature_extractor.extract_features_cv2(image)
        probabilities = self.model.predict_proba(features)[0]

        label_ind = 0
        if probabilities[1] >= self.threshold:
            label_ind = 1

        return self.labels[label_ind], probabilities


if __name__ == '__main__':
    marriage_predictor = MarriagePredictor(threshold=0.5)

    vidcap = cv2.VideoCapture(0)
    while True:
        grabbed_frame, frame = vidcap.read()
        if not grabbed_frame:
            break

        lab, probs = marriage_predictor.predict(frame)

        imutils.text.put_centered_text(frame,
                                       text=lab,
                                       font_face=cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale=1.5,
                                       color=(0, 255, 0),
                                       thickness=2)

        lines = [f'{l}: {100 * p:.2f}%' for l, p in zip(marriage_predictor.labels, probs)]
        lines.append(f'(cutoff @ {100 * marriage_predictor.threshold:.2f}%)')
        prob_text = '\n'.join(lines)

        imutils.text.put_text(frame,
                              text=prob_text,
                              org=(10, 12),
                              font_face=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.5,
                              color=(0, 0, 255),
                              thickness=2)

        cv2.imshow('Marriage Cam', frame)
        key = cv2.waitKey(10)

        if key == 27 or key == ord('q'):
            break
