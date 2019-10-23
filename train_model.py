"""Train logistic regression model on hdf5 features for classification

Modified from:
    https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', default='features.hdf5',
                help='path HDF5 database')
ap.add_argument('-m', '--model', default='model.pickle',
                help='path to output model')
ap.add_argument('-j', '--jobs', type=int, default=-1,
                help='# of jobs to run when tuning hyperparameters')
args = vars(ap.parse_args())

with h5py.File(args['db'], 'r') as db:
    # Train test split (pre-shuffled)
    i = int(db['labels'].shape[0] * 0.9)

    # C decided with sklearn.model_selection.GridSearchCV
    model = LogisticRegression(C=0.1)
    model.fit(db['features'][:i], db['labels'][:i])

    preds = model.predict(db['features'][i:])
    print(classification_report(db['labels'][i:], preds,
                                target_names=db['label_names']))

with open(args['model'], 'wb') as f:
    f.write(pickle.dumps(model))
