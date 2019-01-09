"""Keras-based custom LSTM that classifies into 2 categories"""
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import RMSprop

from tools.helper_keras import keras_train_model

# The Keras Embedding Layer expects a maximum vocabulary size,
# which we know because of the MinMaxScaler scaling to 1, and add a buffer (=2)
VOC_SIZE = 2


def train_model(batch_size, epochs, learning_rate, training_data, training_labels, test_data, test_labels, logdir, model_savename):
    """Creates and trains an instance with given params.

    Args:
        batch_size (int): Batch size for use in training
        epochs (int): How many epochs does the training take
        learn_rate (float): Boosting learning rate (XGB's "eta")
        training_data (numpy.ndarray): Data to train on
        training_labels (list): List of labels corresponding to the training data
        test_data (numpy.ndarray): Data to train on
        test_labels (list): List of labels corresponding to the test data
        logdir (string): In this path all Tensorboard logs will be stored
        model_savename (string): This filename will be used for persisting the trained model

    Returns:
        PredictionEntry: Named tuple with training results

    """
    batch_size = int(batch_size)
    epochs = int(epochs)

    return keras_train_model(
        _create_model(learning_rate),
        epochs, batch_size,
        training_data, training_labels,
        test_data, test_labels,
        logdir,
        model_savename
    )

def _create_model(learning_rate=0.1):
    """Creates a new model with custom learn rate"""
    model = Sequential()
    model.add(Embedding(VOC_SIZE, 32))
    model.add(LSTM(32, name='LSTMnet'))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    return model
