"""Extract features from training images to hd5f db

Modified from:
    https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/
"""
import cv2
import numpy as np
from keras.applications import VGG16
from keras.applications import imagenet_utils


class FeatureExtractor:
    """Wrapper to extract features from images with pre-trained VGG16

    :param batch_size: number of images to extract features from at once
    :ivar batch_size: see param batch_size
    :ivar model: pre-trained VGG16 from keras.applications
    """
    def __init__(self, batch_size=1):
        self.model = VGG16(weights='imagenet', include_top=False)
        self.batch_size = batch_size

    @staticmethod
    def preprocess_cv2_image(cv2_image_bgr):
        """Prepare an OpenCV BGR image for keras.applications.VGG16(weights='imagenet')

        :param cv2_image_bgr: OpenCV style BGR image
        :return: image with attributes prepared for keras VGG16 imagenet model
        """
        cv2_image_bgr = cv2.resize(cv2_image_bgr, (224, 224))
        cv2_image_rgb = cv2.cvtColor(cv2_image_bgr, cv2.COLOR_BGR2RGB).astype('float')
        image_4d = np.expand_dims(cv2_image_rgb, axis=0)
        preprocessed_image = imagenet_utils.preprocess_input(image_4d)

        return preprocessed_image

    def extract_features(self, images, batch_size=None):
        """Extract features from batch of prepped images

        :param images: Array of images prepped for keras.applications.VGG16(weights='imagenet')
        :param batch_size: Number of images to extract features from at once
        :return: Array of features extracted by keras VGG16 imagenet model
        """
        if batch_size is None:
            batch_size = self.batch_size

        features = self.model.predict(images, batch_size=batch_size)
        return features.reshape((features.shape[0], 512 * 7 * 7))

    def extract_features_cv2(self, cv2_image_bgr):
        """Extract VGG16 imagenet features from single OpenCV BGR image

        :param cv2_image_bgr: OpenCV BGR image
        :return: Array of features extracted by keras VGG16 imagenet model
        """
        preprocessed_image = self.preprocess_cv2_image(cv2_image_bgr)
        features = self.extract_features(preprocessed_image, batch_size=1)
        return features.reshape((features.shape[0], 512 * 7 * 7))


if __name__ == '__main__':
    import argparse
    import random
    import os
    from tqdm import tqdm
    import imutils.paths
    from sklearn.preprocessing import LabelEncoder
    from hdf5_dataset_writer import HDF5DatasetWriter

    random.seed(42)
    CLASS_LABELS = {
        'gather_married': 'married',
        'gather_non_married': 'not_married',
    }

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='images',
                    help='path to input dataset')
    ap.add_argument('-o', '--output', default='features.hdf5',
                    help='path to output HDF5 file')
    ap.add_argument('-b', '--batch_size', type=int, default=32,
                    help='batch size of images to be passed through network')
    ap.add_argument('-s', '--buffer_size', type=int, default=1000,
                    help='size of feature extraction buffer')
    args = vars(ap.parse_args())

    all_image_paths = list(imutils.paths.list_images(args['dataset']))
    random.shuffle(all_image_paths)

    class_labels = []
    image_paths = []
    for image_path in tqdm(all_image_paths, desc='Filtering images'):
        dir_label = image_path.split(os.path.sep)[-2]

        try:
            class_label = CLASS_LABELS[dir_label]
        except KeyError:
            continue

        class_labels.append(class_label)
        image_paths.append(image_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(class_labels)

    feature_extractor = FeatureExtractor(batch_size=args['batch_size'])

    dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7),
                                args['output'],
                                data_key='features',
                                buff_size=args['buffer_size'],
                                overwrite=True)
    dataset.store_class_labels(label_encoder.classes_)

    for i in tqdm(range(0, len(image_paths), args['batch_size']), desc='Extracting features'):
        batch_paths = image_paths[i:i + args['batch_size']]
        batch_labels = labels[i:i + args['batch_size']]
        batch_images = []

        # Perform batch feature extraction and add to db
        for (j, image_path) in enumerate(batch_paths):
            image = cv2.imread(image_path)
            image = feature_extractor.preprocess_cv2_image(image)
            batch_images.append(image)

        batch_images = np.vstack(batch_images)
        feats = feature_extractor.extract_features(batch_images)

        dataset.add(feats, batch_labels)

    dataset.close()
