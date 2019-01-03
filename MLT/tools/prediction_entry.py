"""This is a global define for a namded tuple that stores training results."""
from collections import namedtuple


PredictionEntry = namedtuple(
    'PredictionEntry',
    ['test_labels', 'predicted_labels', 'predicted_probabilities', 'training_time']
)
""" A single prediction entry that holds all information of a test run.

Args:
    test_labels: The unmodified, original labels of the test set
    predicted_labels: These are the binary classes that have been predicted (i.e.: 0 or 1)
    predicted_probabilities: A list of probabilities. Each entry represents a value between 0 and 1
    training_time: The time it took for the training to finish
"""
