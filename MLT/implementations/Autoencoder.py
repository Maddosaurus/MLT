"""AutoEncoder pyod implementation based on Aggarwal, C.C. (2015)"""
from pyod.models.auto_encoder import AutoEncoder
from keras.optimizers import adam
import numpy as np

from MLT.tools.helper_pyod import pyod_train_model

# Non-standard impl compared to pyod: Fix the random_state by default and increase verbosity
def train_model(
        training_data, training_labels, test_data, test_labels, full_filename,
        hidden_neurons=None, hidden_activation='relu', output_activation='sigmoid',
        optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2,
        l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
        verbose=2, random_state=42, contamination=0.1, learning_rate=0.001):
    """Created and trains a Autoencoder instance with given params.
    See https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.auto_encoder

    The call is slightly extended in regard to the PyOD version:
    If no hidden_neuron-list is provided, a custom one is generated with a bottleneck of half the feature count.
    Also, if a custom learning rate is provided, an adam optimizer with that lr is used instead of the defaults.

    Returns:
        PredictionEntry: Named tuple with training results
    """

    if hidden_neurons is None:
    # As no neuron layout is given, we need to build one.
    # This model recommends a 1:1 mapping of features <-> neurons, so let's do that.
        no_features = np.size(training_data, 1)
        half_features = int(round(no_features/2))
        hidden_neurons = [no_features, half_features, half_features, no_features]
        print("Built a custom neuron layout: {}".format(hidden_neurons))

    if learning_rate != 1:
        lrate = float(learning_rate) # FIXME: Warning - Float imperfections ahead!
        print("Custom LR. Using adam with lr = {}".format(lrate))
        optimizer = adam(lr=lrate)

    return pyod_train_model(
        _create_model(
            hidden_neurons, hidden_activation, output_activation, optimizer,
            epochs, batch_size, dropout_rate, l2_regularizer, validation_size,
            preprocessing, verbose, random_state, contamination
        ),
        training_data, training_labels,
        test_data, test_labels,
        full_filename
    )


def _create_model(
        hidden_neurons=None, hidden_activation='relu', output_activation='sigmoid',
        optimizer='adam', epochs=100, batch_size=32, dropout_rate=0.2,
        l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
        verbose=2, random_state=42, contamination=0.1):
    """(Internal helper) Created an Autoencoder instance"""

    autoenc = AutoEncoder(
        hidden_neurons=hidden_neurons,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        optimizer=optimizer,
        epochs=int(epochs),
        batch_size=int(batch_size),
        dropout_rate=dropout_rate,
        l2_regularizer=l2_regularizer,
        validation_size=validation_size,
        preprocessing=preprocessing,
        verbose=verbose,
        random_state=random_state,
        contamination=contamination
    )

    return autoenc
