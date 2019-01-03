"""Basic scikit implementation of a Random Forest Classifier"""
from sklearn.ensemble import RandomForestClassifier

from tools.helper_sklearn import sklearn_train_model

def train_model(n_estimators, max_depth, training_data, training_labels, test_data, test_labels, model_savename):
    """Creates and trains a XGBoost sklearn instance with given params.

    Args:
        n_estimators (int): Number of estimators to use
        max_depth (int): Maximum tree depth for individual trees
        training_data (numpy.ndarray): Data to train on
        training_labels (list): List of labels corresponding to the training data
        test_data (numpy.ndarray): Data to train on
        test_labels (list): List of labels corresponding to the test data
        full_filename (string): This filename will be used for persisting the trained model

    Returns:
        PredictionEntry: Named tuple with training results
    """
    return sklearn_train_model(
        _createModel(n_estimators, max_depth),
        training_data, training_labels,
        test_data, test_labels,
        model_savename
    )

def _createModel(n_estimators=100, max_depth=None):
    """Creates a new model with custom ammount of estimators and maximum tree depth"""
    if max_depth is not None:
        max_depth = int(max_depth)
        if max_depth == 0:
            max_depth = None
    n_estimators = int(n_estimators)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,  # Number of trees in the forest
        max_depth=max_depth,        # Maximum depth of the tree
        oob_score=False,            # Use oob to estimate generalization acc
        random_state=0,             # Fixed init state
        n_jobs=-1,                  # Set number of jobs = CPU cores
    )
    return rf
