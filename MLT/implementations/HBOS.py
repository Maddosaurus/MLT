"""HBOS pyod implementation based on Goldstein and Dengel (2012)"""
from pyod.models.hbos import HBOS

from MLT.tools.helper_pyod import pyod_train_model

def train_model(n_bins, alpha, tol, contamination, training_data, training_labels, test_data, test_labels, full_filename):
    """Created and trains a HBOS instance with given params

    Args:
        n_bins (int, optional (default=10)): The number of bins
        alpha (float in (0, 1), optional (default=0.1)): The regularizer for preventing overflow
        tol (float in (0, 1), optional (default=0.1)): The parameter to decide the flexibility while dealing the samples falling outside the bins.
        training_data (numpy.ndarray or Pandas.DataFrame): Data to train on
        training_labels (list): List of labels corresponding to the training data - can be left empty for unsupervised learning
        test_data (numpy.ndarray or Pandas.DataFrame): Data to train on
        test_labels (list): List of labels corresponding to the test data

    Returns:
        PredictionEntry: Named tuple with training results
    """
    return pyod_train_model(
        _create_model(n_bins, alpha, tol, contamination),
        training_data, training_labels,
        test_data, test_labels,
        full_filename
    )


def _create_model(n_bins=10, alpha=0.1, tol=0.1, contamination=0.1):
    """(Internal helper) Create a HBOS instance"""
    n_bins = int(n_bins)

    hbos = HBOS(
        n_bins=n_bins,
        alpha=alpha,
        tol=tol,
        contamination=contamination
    )

    print('Created Model: {}'.format(hbos))

    return hbos
