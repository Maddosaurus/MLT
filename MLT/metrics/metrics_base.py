"""Utility module for various basic metric functions.

These functions all take stats_data and transforms these to a list of target metrics for all folds.
"""
from datetime import timedelta
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def calc_acc(prediction_data):
    """Calculate basic accuracy for a given list of prediction entries"""
    all_acc = []
    for pred in prediction_data:
        all_acc.append(accuracy_score(pred.test_labels, pred.predicted_labels))
    return all_acc

def calc_precision(prediction_data):
    """Calculate the precision for a given list of predictions."""
    all_prec = []
    for pred in prediction_data:
        all_prec.append(precision_score(pred.test_labels, pred.predicted_labels))
    return all_prec

def calc_recall(prediction_data):
    """Calculate the recall for a given list of predictions."""
    all_rec = []
    for pred in prediction_data:
        all_rec.append(recall_score(pred.test_labels, pred.predicted_labels))
    return all_rec

def calc_fbeta_binary(prediction_data, beta):
    """Calculate fß score for a given list of prediction entries"""
    all_fb = []
    for pred in prediction_data:
        all_fb.append(fbeta_score(pred.test_labels, pred.predicted_labels, beta))
    return all_fb

def sum_training_times(stats_data):
    """Sum all training times of all folds"""
    final = timedelta(0)
    for entry in stats_data:
        final = final + entry.training_time
    return str(final)

def calc_mean_training_time(stats_data):
    """Calculate the mean traning time over all folds"""
    times = []
    for entry in stats_data:
        times.append(entry.training_time)
    return str(np.mean(times))
