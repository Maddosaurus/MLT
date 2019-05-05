"""iForest implementation by pyod based on scikit-learn"""
from pyod.models.iforest import IForest

from MLT.tools.helper_pyod import pyod_train_model

def train_model(
    training_data, training_labels, test_data, test_labels, full_filename,
    n_estimators=100, contamination=0.1, max_features=1.0, bootstrap=False):
    """Created and trains an Isolation Forest instance with given params

    Args:
        n_estimators (int, optional (default=100)): The number of base estimators in the ensemble.
        contamination (float in (0., 0.5), optional (default=0.1)): The amount of contamination of the data set
        max_features (int or float, optional (default=1.0)): The number of features to draw from X to train each base estimator.
        bootstrap (boolean, optional (default=False)): If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.

    Returns:
        PredictionEntry: Named tuple with training results
    """

    return pyod_train_model(
        _create_model(n_estimators, contamination, max_features, bootstrap),
        training_data, training_labels,
        test_data, test_labels,
        full_filename
    )


def _create_model(n_estimators=100, contamination=0.1, max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0):
    """(Internal helper) Creates an Isolation Forest instance"""
    n_estimators = int(n_estimators)
    contamination = float(contamination)
    max_features = float(max_features)
    bootstrap = bool(bootstrap)

    forest = IForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    print('Created Model: {}'.format(forest))

    return forest
