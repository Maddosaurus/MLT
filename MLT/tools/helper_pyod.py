"""Utility functions for pyod-related implementations"""
import os
from datetime import datetime
from sklearn.externals import joblib

from MLT.tools import prediction_entry as pe

def pyod_train_model(model, training_data, training_labels, test_data, test_labels, model_savename):
    """Train the given model with data and predict the run"""
    starttime = datetime.now()

    if training_labels is None:
        model.fit(training_data)
    else:
        model.fit(training_data, training_labels)

    finishtime = datetime.now()
    runtime = finishtime - starttime

    # predict the run
    test_predictions = model.predict(test_data)
    test_predictions_probabilities = model.predict_proba(test_data)[:, 1]

    pyod_persist_model(model, model_savename)
    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)


def pyod_persist_model(model, model_savename):
    """Save a scikit model to disk"""
    joblib.dump(model, model_savename + '.pkl')

def pyod_load_model(dirpath, modelname):
    """Load a scikit model from disk"""
    model_path = os.path.join(dirpath, modelname)
    return joblib.load(model_path)

def pyod_load_modellist(model_filenames, model_path):
    """Load a list of scikit models from disk from given path"""
    loaded_models = []
    for model_fname in model_filenames:
        filename_wo_ext = os.path.splitext(model_fname)[0]
        loaded_models.append(
            (
                filename_wo_ext,
                pyod_load_model(model_path, model_fname)
            )
        )
    return loaded_models

def predict_pyod(single_model, test_data, test_labels):
    """Only predict a model without fitting it"""
    starttime = datetime.now()

    test_predictions = single_model.predict(test_data)
    test_predictions_probabilities = single_model.predict_proba(test_data)[:, 1]

    finishtime = datetime.now()
    runtime = finishtime - starttime

    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)