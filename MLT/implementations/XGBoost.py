"""XGBoost scikit implementation based on https://xgboost.readthedocs.io/en/latest/"""
from xgboost import XGBClassifier

from tools.helper_sklearn import sklearn_train_model

def train_model(n_estimators, max_depth, learning_rate, training_data, training_labels, test_data, test_labels, full_filename):
    """Creates and trains a XGBoost sklearn instance with given params

    Args:
        n_estimators (int):    Number of estimators to use
        max_depth (int):       Maximum tree depth for base learners
        learning_rate (float): Boosting learning rate (XGB's "eta")
        training_data (numpy.ndarray): Data to train on
        training_labels (list): List of labels corresponding to the training data
        test_data (numpy.ndarray): Data to train on
        test_labels (list): List of labels corresponding to the test data
        full_filename (string): This filename will be used for persisting the trained model

    Returns:
        PredictionEntry: Named tuple with training results
    """
    return sklearn_train_model(
        _create_model(n_estimators, max_depth, learning_rate),
        training_data, training_labels,
        test_data, test_labels,
        full_filename
    )

def _create_model(n_estimators=100, max_depth=3, learning_rate=0.1):
    """(Internal helper) Creates a scikit-learn-compatible XGBoost instance"""
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    xgb = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_jobs=-1,                      # Set number of jobs = CPU cores
        random_state=0                  # Fixed init state
    )

    return xgb
