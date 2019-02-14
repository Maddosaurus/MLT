import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('./MLT'))

from MLT.tools import prediction_entry as pe
from MLT.metrics import metrics
from MLT.datasets import CIC
from MLT.datasets import NSL


def baseline_NSL16():
    train_data, test_data, train_labels, test_labels = NSL.get_NSL_16class()

    # generate list that exclusively hold true or false
    label_count = len(test_labels)
    all_true = np.ones(label_count, np.int)

    at = pe.PredictionEntry(test_labels, all_true, [], [])

    result_path = os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'NSL_KDD')

    metrics.calc_metrics_wo_probabilities([at], "baseline_true", result_path)


def baseline_CIC16():
    train_data, test_data, train_labels, test_labels = CIC.get_CIC_Top16()

    # generate list that exclusively hold true or false
    label_count = len(test_labels)
    all_true = np.ones(label_count, np.int)

    at = pe.PredictionEntry(test_labels, all_true, [], [])

    result_path = os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'CICIDS2017')

    metrics.calc_metrics_wo_probabilities([at], "baseline_true", result_path)


if __name__ == '__main__':
    baseline_NSL16()
    baseline_CIC16()