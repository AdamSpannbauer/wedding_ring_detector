"""Train logistic regression model on hdf5 features for classification

Modified from:
    https://gurus.pyimagesearch.com/topic/transfer-learning-example-dogs-and-cats/
"""
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import h5py


def train_model(h5py_db, model_output='model.pickle'):
    """Train logistic regression classifier

    :param h5py_db: path to HDF5 database containing 'features', 'labels', & 'label_names'
    :param model_output: path to save trained model to using pickle
    :return: None; output is written to `model_output`
    """
    # Train test split (assumed to be pre-shuffled)
    i = int(h5py_db['labels'].shape[0] * 0.7)

    # C decided with sklearn.model_selection.GridSearchCV
    model = LogisticRegression(C=0.1)
    model.fit(h5py_db['features'][:i], h5py_db['labels'][:i])

    preds = model.predict(h5py_db['features'][i:])
    print(classification_report(h5py_db['labels'][i:], preds,
                                target_names=h5py_db['label_names']))

    with open(model_output, 'wb') as f:
        f.write(pickle.dumps(model))


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--db', default='features.hdf5',
                    help='path HDF5 database')
    ap.add_argument('-m', '--model', default='model.pickle',
                    help='path to output model')
    args = vars(ap.parse_args())

    with h5py.File(args['db'], 'r') as db:
        train_model(db, args['model'])
