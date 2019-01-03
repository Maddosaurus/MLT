"""Collection of misc tools that don't fit in a standalone module"""
import os
import json
import pickle
import natsort
import numpy as np
from datetime import datetime

def list_files(dirpath, fname_start):
    """List all files in a folder that start with the given string."""
    filelist = (filename for filename in os.listdir(dirpath) if filename.startswith(fname_start))
    return natsort.natsorted(filelist)


def list_folders(dirpath):
    """List all subfolders in a given path"""
    folderlist = [f for f in os.listdir(dirpath) if not os.path.isfile(os.path.join(dirpath, f))]
    return natsort.natsorted(folderlist)


def create_dir(dirpath):
    """Create the specified path if it is not existing."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def write_to_json(full_path_with_name, data):
    """JSON dump the given file to disk at the given path"""
    with open(full_path_with_name, 'w') as mjson:
        json.dump(data, mjson)


def read_from_json(full_path_with_name):
    """Read from an arbitrary JSON and return the structure"""
    with open(full_path_with_name, 'r') as mjson:
        return json.load(mjson)


def write_to_pickle(full_path_with_name, data):
    """Pickle the given file to disk at the given path"""
    with open(full_path_with_name, 'wb') as pickle_handle:
        pickle.dump(data, pickle_handle)


def read_from_pickle(full_path_with_name):
    """PRead from pickle at given location"""
    with open(full_path_with_name, 'rb') as pickle_handle:
        return pickle.load(pickle_handle)


def write_call_params(args, result_path):
    """Write the parametes with wich MLT has been called to a txt file in the result path"""
    parampath = os.path.join(result_path, 'call_parameters.txt')
    with open(parampath, 'w') as paramwriter:
        paramwriter.write('MLT call params:\n\n')
        paramwriter.write(str(args))
        paramwriter.close()


def save_np_to_disk(stats_dataframe, filename, result_path):
    """Save a given dataframe as binary numpy pickle to disk"""
    filepath = os.path.join(result_path, filename + '.npy')
    np.save(filepath, stats_dataframe)


def save_metrics_to_disk(metrics_array, modelname, result_path):
    """Save a given metric array as json to disk"""
    filepath = os.path.join(result_path, modelname +'_metrics.json')
    with open(filepath, 'w') as mjson:
        json.dump(metrics_array, mjson)


def save_results_to_disk(stats_data, filename, result_path):
    """save the full results for a given model as json to disk"""
    filepath = os.path.join(result_path, filename + '_results.json')
    dictlist = []
    for stat in stats_data:
        dstat = {}
        dstat['test_labels'] = stat.test_labels.ravel().tolist()
        dstat['predicted_labels'] = stat.predicted_labels.ravel().tolist()
        dstat['training_time'] = str(stat.training_time)
        if isinstance(stat.predicted_probabilities, list):
            dstat['predicted_probabilities'] = stat.predicted_probabilities
        else:
            dstat['predicted_probabilities'] = stat.predicted_probabilities.ravel().tolist()
        dictlist.append(dstat) # convert to JSON-compatible dict

    with open(filepath, 'w') as jresult:
        json.dump(dictlist, jresult)


def load_fold_indices(path):
    """Load the stard and end indices of the test set for every fold."""
    filename = os.path.join(path, 'dataset_fold_indices.json')
    with open(filename, 'r') as handle:
        parsed = json.load(handle)
        return json.dumps(parsed['short'], indent=4)


def load_result(path, modelname):
    """Load the metrics for a given model in the given path."""
    filename = os.path.join(path, modelname + '_metrics.json')
    with open(filename, 'r') as handle:
        parsed = json.load(handle)
        return json.dumps(parsed, indent=4)


def load_results_from_disk(path, modelname):
    """Load the full result json for the given model from the path."""
    filename = os.path.join(path, modelname + '_results.json')
    with open(filename, 'r') as handle:
        parsed = json.load(handle)
        return parsed

def prepare_folders(runner_name):
    """Creates all the folders needed for a test run
    
    Args:
        runner_name (string): Name of the calling runner. Will be the base name for results

    Returns
        result_path (string): The full path where results can be stored
    """

    runtime_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = (
        os.path.join(
            os.path.dirname(__file__), '..', 'results', runner_name, runtime_date
        )
    )
    model_savepath = os.path.join(result_path, 'models')

    # Housekeeping! Create the result path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    
    return result_path
