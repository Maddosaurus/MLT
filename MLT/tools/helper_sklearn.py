"""Utility functions for scikit-learn-realted implementations"""
import os
from datetime import datetime
from sklearn.externals import joblib

from MLT.tools import prediction_entry as pe

def sklearn_train_model(model, training_data, training_labels, test_data, test_labels, model_savename):
    """Train the given model with data and predict the run"""
    starttime = datetime.now()
    model.fit(training_data, training_labels)
    finishtime = datetime.now()
    runtime = finishtime - starttime

    #predict the run
    test_predictions = model.predict(test_data)
    test_predictions_probabilities = model.predict_proba(test_data)[:, 1]
    # proba[:,1] returns just 1 of 2 columns. As they always add up, this is enough!

    sklearn_persist_model(model, model_savename)
    # append all this to a dataframe / JSON / whatever and
    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)


def sklearn_persist_model(model, model_savename):
    """Save a scikit model to disk"""
    joblib.dump(model, model_savename + '.pkl')

def sklearn_load_model(dirpath, modelname):
    """Load a scikit model from disk"""
    model_path = os.path.join(dirpath, modelname)
    return joblib.load(model_path)

def sklearn_load_modellist(model_filenames, model_path):
    """Load a list of scikit models from disk from given path"""
    loaded_models = []
    for model_fname in model_filenames:
        filename_wo_ext = os.path.splitext(model_fname)[0]
        loaded_models.append(
            (
                filename_wo_ext,
                sklearn_load_model(model_path, model_fname)
            )
        )
    return loaded_models


def predict_scikit(single_model, test_data, test_labels):
    """Only predict a model without fitting it"""
    starttime = datetime.now()

    test_predictions = single_model.predict(test_data)
    test_predictions_probabilities = single_model.predict_proba(test_data)[:, 1]

    finishtime = datetime.now()
    runtime = finishtime - starttime

    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)
