"""Utility functions for Keras-realted implementations"""
import os
from datetime import datetime
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
import keras.models
import keras.backend as K

from MLT.tools import prediction_entry as pe

# Shut off TensorFlow debug info
# see https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def keras_train_model(model, epochs, batch_size, training_data, training_labels, test_data, test_labels, logdir, model_savename):
    """Train the given model with data and predict the run."""
    starttime = datetime.now()
    history = model.fit(
        training_data, training_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_data, test_labels),
        callbacks=[
            TensorBoard(log_dir=logdir)
        ]
    )
    finishtime = datetime.now()
    runtime = finishtime - starttime

    test_predictions_probabilities = model.predict(test_data)

    test_predictions = test_predictions_probabilities.argmax(axis=-1)
    test_predictions_probabilities = test_predictions_probabilities[:, 1]

    keras_persist_model(model, model_savename)

    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)


def keras_train_model_adaptive(model, epochs, batch_size, training_data, training_labels, test_data, test_labels, logdir, model_savename):
    """Train the given model with data and predict the run.

    This training reduces the learning rate on a fixed base every 30 epochs to 10% of the original value."""

    # see https://github.com/keras-team/keras/issues/888#issuecomment-150849433
    def adaptive_lr_scheduler(epoch):
        if (epoch > 0) and (epoch % 30 == 0):
            old_lr = K.get_value(model.optimizer.lr)
            new_lr = old_lr * 0.1
            print("Set LR to {:6.5f}".format(new_lr))
            K.set_value(model.optimizer.lr, new_lr)
        print("Current LR: {:6.5f}".format(K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)


    starttime = datetime.now()
    history = model.fit(
        training_data, training_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_data, test_labels),
        callbacks=[
            TensorBoard(log_dir=logdir),
            LearningRateScheduler(adaptive_lr_scheduler)
        ]
    )
    finishtime = datetime.now()
    runtime = finishtime - starttime

    test_predictions_probabilities = model.predict(test_data)

    test_predictions = test_predictions_probabilities.argmax(axis=-1)
    test_predictions_probabilities = test_predictions_probabilities[:, 1]
    # proba[:,1] returns just 1 of 2 columns. As they always add up, this is enough!

    keras_persist_model(model, model_savename)

    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)

def keras_persist_model(model, model_savename):
    """Save the full model to disk."""
    model.save(model_savename + '.h5')

def keras_load_model(full_path):
    """Load a single model from given path"""
    return keras.models.load_model(full_path)

def keras_load_modellist(model_filenames, model_path):
    """Load a list of models from a path"""
    loaded_models = []
    for model_fname in model_filenames:
        loaded_models.append(
            keras_load_model(os.path.join(model_path, model_fname))
        )
    return loaded_models


def predict_keras(single_model, test_data, test_labels):
    """Only predict a model without training it."""
    starttime = datetime.now()

    test_predictions_probabilities = single_model.predict(test_data)
    test_predictions = test_predictions_probabilities.argmax(axis=-1)

    finishtime = datetime.now()
    runtime = finishtime - starttime

    test_predictions_probabilities = test_predictions_probabilities[:, 1]

    return pe.PredictionEntry(test_labels, test_predictions, test_predictions_probabilities, runtime)
