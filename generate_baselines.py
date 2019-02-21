import os
import sys
import numpy as np
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.abspath('./MLT'))

from MLT.tools import prediction_entry as pe
from MLT.metrics import metrics
from MLT.datasets import CIC
from MLT.datasets import NSL



def baseline_NSL16():
    print("NSL Baseline")
    train_data, test_data, train_labels, test_labels = NSL.get_NSL_16class()
    result_path = os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'NSL_KDD')

    _baseline_data(train_data, test_data, train_labels, test_labels, result_path)


def baseline_CIC16():
    print("CIC Baseline")
    train_data, test_data, train_labels, test_labels = CIC.get_CIC_Top16()
    result_path = os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'CICIDS2017')

    _baseline_data(train_data, test_data, train_labels, test_labels, result_path)


def _baseline_data(train_data, test_data, train_labels, test_labels, result_path):
    at = _gen_single_pe(test_labels)
    metrics.calc_metrics_wo_probabilities([at], "baseline_test", result_path)

    at = _gen_single_pe(train_labels)
    metrics.calc_metrics_wo_probabilities([at], "baseline_train_full", result_path)

    td = train_data.values
    fold_results =[]
    kfold = KFold(n_splits=10)
    for train, test in kfold.split(td):
        fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = td[train], td[test], train_labels[train], train_labels[test]
        fold_results.append(_gen_single_pe(fold_test_labels))
    print("Debug: I now have {} measurements. Averaging... ".format(len(fold_results)))
    metrics.calc_metrics_wo_probabilities([at], "baseline_train_10fold", result_path)


def _gen_single_pe(labels):
    label_count = len(labels)
    all_true = np.ones(label_count, np.int)
    at = pe.PredictionEntry(labels, all_true, [], [])
    return at


if __name__ == '__main__':
    baseline_NSL16()
    baseline_CIC16()