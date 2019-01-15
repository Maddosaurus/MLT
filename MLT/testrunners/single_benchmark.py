# pylint: disable=C0103,C0301,C0326,W0703
"""This runner implements the main benchmark for qualitative analysis based on the full training and test sets."""
import os
import warnings

from MLT.datasets import NSL
from MLT.datasets import CIC_6class

from MLT.implementations import XGBoost
from MLT.implementations import RandomForest
from MLT.implementations import LSTM_2_Multiclass
from MLT.implementations import HBOS

from MLT.metrics import metrics
from MLT.tools import dataset_tools
from MLT.tools import result_mail
from MLT.tools import toolbelt

# supress deprecation warning. sklearn is currently built against an older numpy version.
warnings.filterwarnings(
    module='sklearn*', action='ignore',
    category=DeprecationWarning,
    message='The truth value of an empty array is ambiguous*'
)

# supress future warning, as this is in the responsibility of pyod.
warnings.filterwarnings(
    module='scipy*', action='ignore',
    category=FutureWarning,
    message='Using a non-tuple sequence for multidimensional*'
)


def run_NSL(args):
    """Run the benchmark for a six feature subset of NSL_KDD.

    Args:
        args (argparse.Namespace): Argument Namespace containing all parameters for the test run

    Returns:
        result_path (string): Full path where the results can be found
    """
    if args.NSL6:
        kdd_train_data, kdd_test_data, kdd_train_labels, kdd_test_labels = NSL.get_NSL_6class()
        result_path = toolbelt.prepare_folders('NSL_6class_single')
    elif args.NSL16:
        kdd_train_data, kdd_test_data, kdd_train_labels, kdd_test_labels = NSL.get_NSL_16class()
        result_path = toolbelt.prepare_folders('NSL_16class_single')

    model_savepath = os.path.join(result_path, 'models')
    # also, save parameters with which the runner was called
    toolbelt.write_call_params(args, result_path)

    # convert to numpy.ndarrays...
    kdd_train_data = kdd_train_data.values
    kdd_test_data = kdd_test_data.values

    # ... and run the benchmark!
    return run_benchmark(kdd_train_data, kdd_train_labels, kdd_test_data, kdd_test_labels, result_path, model_savepath, args)


def CIC_6c(args, stratified=True):
    """Run the benchmark for a (stratified or randomized) six feature subset of CICIDS2017.

    Args:
        args (argparse.Namespace): Argument Namespace containing all parameters for the test run
        stratified (boolean, (optional) default=True): Whether to use the straified or the randomized test data set

    Returns:
        result_path (string): Full path where the results can be found
    """
    cic_runnername = "CIC_6class_single"

    if stratified:
        cic_train_data, cic_test_data, cic_train_labels, cic_test_labels = CIC_6class.get_CIC_6class_stratified()
        cic_runnername += '-stratified'
    else:
        cic_train_data, cic_test_data, cic_train_labels, cic_test_labels = CIC_6class.get_CIC_6class_randomized()
        cic_runnername += '-randomized'

    result_path = toolbelt.prepare_folders(cic_runnername)
    model_savepath = os.path.join(result_path, 'models')

    # also, save parameters with which the runner was called
    toolbelt.write_call_params(args, result_path)

    # convert to numpy.ndarrays
    cic_train_data = cic_train_data.values
    cic_test_data = cic_test_data.values

    # and run the benchmark!
    return run_benchmark(cic_train_data, cic_train_labels, cic_test_data, cic_test_labels, result_path, model_savepath, args)



def run_benchmark(train_data, train_labels, test_data, test_labels, result_path, model_savepath, args):
    """Run the full benchmark.

    As this is the full benchmark, it needs a train and a test partition.
    Besides that, it is mostly similar to the kfold_runner.

    Args:
        train_data (numpy.ndarray): Training partition
        train_labels (numpy.ndarray): According labels for supervised learning
        test_data (numpy.ndarray): Training partition
        test_labels (numpy.ndarray): According labels for supervised learning
        result_path (str): Where to save the results
        model_savepath (str): Where to store the trainned models
        args (argparse.Namespace): Parsed CMD arguments that contain all the switches and settings

    Returns:
        result_path (str): The path where to find the final results
    """
    withXGBoost      = args.XGBoost
    withRandomForest = args.RandomForest
    withLSTM2        = args.LSTM2
    withHBOS         = args.HBOS

    xgboost_stats       = []
    random_forest_stats = []
    lstm2_stats         = []
    hbos_stats          = []

    # normalize and scale the data splits
    train_data, test_data = dataset_tools.normalize_and_scale(train_data, test_data)

    if args.unsupervised:
        train_labels = None # Pass empty train labels

    if withXGBoost:
        print("Training XGBoost")
        full_filename = os.path.join(model_savepath, "XGBoost")
        xgb_train_pass = XGBoost.train_model(
            withXGBoost[0], # n_estimators
            withXGBoost[1], # max_depth
            withXGBoost[2], # learning_rate
            train_data,
            train_labels,
            test_data,
            test_labels,
            full_filename
        )
        xgboost_stats.append(xgb_train_pass)

    if withRandomForest:
        print("Training Random Forest")
        full_filename = os.path.join(model_savepath, "RandomForest")
        random_forest_pass = RandomForest.train_model(
            withRandomForest[0], # n_estimators
            withRandomForest[1], # max_depth
            train_data,
            train_labels,
            test_data,
            test_labels,
            full_filename
        )
        random_forest_stats.append(random_forest_pass)

    if withLSTM2:
        print("Training 2-Class LSTM")
        tensorboard_logir = os.path.join(result_path, 'LSTM2C')
        mode_savepath = os.path.join(model_savepath, 'LSTM2C')
        lstm2_pass = LSTM_2_Multiclass.train_model(
            withLSTM2[0], # batch_size
            withLSTM2[1], # epochs
            withLSTM2[2], # learning_rate
            train_data,
            train_labels,
            test_data,
            test_labels,
            tensorboard_logir,
            mode_savepath
        )
        lstm2_stats.append(lstm2_pass)

    if withHBOS:
        print("Training HBOS")
        full_filename = os.path.join(model_savepath, "HBOS")
        random_forest_pass = HBOS.train_model(
            withHBOS[0], # n_bins
            withHBOS[1], # alpha
            withHBOS[2], # tol
            train_data,
            train_labels,
            test_data,
            test_labels,
            full_filename
        )
        hbos_stats.append(random_forest_pass)


    try:
        if withXGBoost:
            metrics.calc_metrics_and_save_to_disk(xgboost_stats, 'XGBoost', result_path)
    except Exception:
        print('Ran into exception while saving XGBoost results to disk')

    try:
        if withRandomForest:
            metrics.calc_metrics_and_save_to_disk(random_forest_stats, 'RandomForest', result_path)
    except Exception:
        print('Ran into exception while saving Random Forest results to disk')

    try:
        if withLSTM2:
            metrics.calc_metrics_and_save_to_disk(lstm2_stats, 'LSTM2C', result_path)
    except Exception:
        print('Ran into exception while saving LSTM2C results to disk')

    try:
        if withHBOS:
            metrics.calc_metrics_and_save_to_disk(hbos_stats, 'HBOS', result_path)
    except Exception:
        print('Ran into exception while saving HBOS results to disk')


    if args.ResultMail:
        result_mail.prepare_and_send_results(result_path, args)

    full_respath = os.path.abspath(result_path)
    print('Results have ben saved to {}'.format(full_respath))

    # Return the full path for other runners to use
    return full_respath
