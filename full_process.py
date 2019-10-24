import argparse
import imutils.paths
import h5py
from gather_images import gather_images, try_int, prompt_labels
from extract_features import extract_features
from train_model import train_model
from video_classify import video_classify

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train_input', type=try_int, default=0,
                help='Path to input video to capture training data from. '
                     '(can be number to indicate web cam; see cv2.VideoCapture docs)')
ap.add_argument('-o', '--train_output', default='images',
                help='Main dir for training images to be saved to. '
                     '(they will saved to a subdir defined by FLAGS dict)')
ap.add_argument('-n', '--n_classes', type=int, default=5,
                help='Number of classes to define')
ap.add_argument('-d', '--feature_db', default='features.hdf5',
                help='path to output HDF5 file')
ap.add_argument('-m', '--model_output', default='model.pickle',
                help='path to output model to')
ap.add_argument('-b', '--batch_size', type=int, default=32,
                help='batch size of images to extract features from at once')
ap.add_argument('-s', '--buffer_size', type=int, default=1000,
                help='size of feature extraction buffer')
args = vars(ap.parse_args())

# Gather training images from video
labels, _ = prompt_labels(args['n_classes'])
gather_images(output_dir=args['train_output'],
              labels=labels,
              video_capture=args['train_input'])

# Extract features from training images with imagenet
im_paths = list(imutils.paths.list_images(args['train_output']))
extract_features(im_paths, args['feature_db'], args['batch_size'], args['buffer_size'])

# Train logistic regression model
with h5py.File(args['feature_db'], 'r') as db:
    labels = list(db['label_names'])
    train_model(db, args['model_output'])

video_classify(labels, args['model_output'], args['train_input'])
