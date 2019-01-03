"""Base module that functions as single point of entry and facade for metrics generation.

It's main function awaits a stats_data list, which is a list of named tuples,
If you're doing a 10-fold crossvalidation, the list contains 10 entries.
A Single entry looks like this:
('PredictionEntry', ['test_labels', 'predicted_labels', 'predicted_probabilities', 'training_time'])
"""
import json
import os
import numpy as np

from metrics import metrics_base
from metrics import metrics_cm
from metrics import metrics_roc
from tools import toolbelt
from tools import result_helper

def calc_metrics_and_save_to_disk(stats_data, modelname, result_path):
    """Main entry and facade for metrics generation.

    There is a service function called at the end.
    The separation was needed, as probas aren't always available.
    The additional stuff that needs probas is done here, the rest in the service function.
    Args:
        stats_data (list): List of PredictionEntry named tuples that contain the full fold info
        modelname (string): Name of the model that is evaluated. This determines the filename
        result_path (string): Where the generated metrics will be saved
    """

    metrics_data = {}
    metrics_data['training_time_sum'] = metrics_base.sum_training_times(stats_data)
    metrics_data['training_time_mean'] = metrics_base.calc_mean_training_time(stats_data)

    auc = metrics_roc.calc_auc(stats_data)
    metrics_roc.generate_avg_roc_to_disk(stats_data, modelname, result_path)
    append_to_metrics_data(metrics_data, 'auc', auc)

    calc_metrics_wo_probabilities(stats_data, modelname, result_path, metrics_data)


def calc_metrics_wo_probabilities(stats_data, modelname, result_path, metrics_data=None):
    """Service function for moments where no predicted_probas are available.
    
    Args:
        stats_data (list): List of PredictionEntry named tuples that contain the full fold info
        modelname (string): Name of the model that is evaluated. This determines the filename
        result_path (string): Where the generated metrics will be saved
        metrics_data (dict): (Optional) Prefilled dict with additional stats
    """
    print('\nCalc stats for {}:'.format(modelname))

    if metrics_data is None:
        metrics_data = {}

    acc = metrics_base.calc_acc(stats_data)
    precision = metrics_base.calc_precision(stats_data)
    recall = metrics_base.calc_recall(stats_data)
    f1_score = metrics_base.calc_fbeta_binary(stats_data, 1.0)
    cm_arr = metrics_cm.calc_cm(stats_data)

    metrics_cm.generate_all_cm_to_disk(cm_arr, modelname, result_path)

    # serialize all the metrics to a single file
    append_to_metrics_data(metrics_data, 'acc', acc)
    append_to_metrics_data(metrics_data, 'precision', precision)
    append_to_metrics_data(metrics_data, 'recall', recall)
    append_to_metrics_data(metrics_data, 'f1_score', f1_score)

    # print(json.dumps(metrics_data, indent=4))
    toolbelt.save_metrics_to_disk(metrics_data, modelname, result_path)

    # save the full array with all results
    toolbelt.save_results_to_disk(stats_data, modelname, result_path)

    metrics_cm.save_cm_arr_to_disk(cm_arr, modelname, result_path)

    result_helper.list_single_score(modelname, result_path)
    print('End of stats for {}\n'.format(modelname))


# metrics_data is call by reference, so no need for a return!
def append_to_metrics_data(metrics_data, metric, results):
    """Append the results of given metric as new entry to metrics_data"""
    metrics_data[metric] = {}
    metrics_data[metric]['mean'] = np.mean(results)
    metrics_data[metric]['var'] = np.var(results)
    metrics_data[metric]['sd'] = np.std(results)
