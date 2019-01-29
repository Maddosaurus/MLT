"""This runner implements the benchmark with a configurable number of k-Folds for crossvalidation"""
import os
import warnings
import json
from sklearn.model_selection import KFold

from MLT.implementations import Autoencoder, HBOS, IsolationForest, LSTM_2_Multiclass, RandomForest, XGBoost

from MLT.metrics import metrics
from MLT.tools import dataset_tools, result_mail

# supress deprecation warning. sklearn is currently built against an older numpy version.
warnings.filterwarnings(
    module='sklearn*', action='ignore',
    category=DeprecationWarning,
    message='The truth value of an empty array is ambiguous'
)

def run_benchmark(candidate_data, candidate_labels, result_path, model_savepath, args):
    """Run the k-fold benchmark itself.

    Note the absence of train- and test-partitions.
    As this is a crossvalidation run, the test partition is not to be touched!

    Keyword arguments:
    candidate_data   -- Training data with 6 features
    candidate_labels -- According labels for supervised learning
    result_path      -- Where to save the results
    args             -- Parsed CMD arguments that contain all the switches and settings
    """
    kfold_count      = args.kfolds
    withXGBoost      = args.XGBoost
    withRandomForest = args.RandomForest
    withLSTM2        = args.LSTM2
    withHBOS         = args.HBOS
    withAutoEnc      = args.AutoEncoder
    withIForest      = args.IsolationForest

    xgboost_stats       = []
    random_forest_stats = []
    lstm2_stats         = []
    hbos_stats          = []
    autoenc_stats       = []
    iforest_stats       = []

    fold_indices          = {}
    fold_indices['short'] = {}
    fold_indices['full']  = {}

    fold_counter = 1

    # k-fold Crossval. Split into k-1 training and 1 test part - repeat k times.
    kfold = KFold(n_splits=kfold_count)
    for train, test in kfold.split(candidate_data):
        fold_indices['short'][fold_counter] = {}
        fold_indices['short'][fold_counter]['test_indices'] = str(test)
        fold_indices['full'][fold_counter] = {}
        fold_indices['full'][fold_counter]['train_indices'] = list(train.tolist())
        fold_indices['full'][fold_counter]['test_indices'] = list(test.tolist())

        fold_train_data, fold_test_data, fold_train_labels, fold_test_labels = candidate_data[train], candidate_data[test], candidate_labels[train], candidate_labels[test]

        if args.unsupervised:
            fold_train_labels = None # Pass empty train labels

        # normalize and scale the data splits
        fold_train_data, fold_test_data = dataset_tools.normalize_and_scale(fold_train_data, fold_test_data)

        # now fit the models
        print('\nBeginning training pass {:2d}/{}'.format(fold_counter, kfold_count))

        if withXGBoost:
            print("Training XGBoost")
            full_filename = os.path.join(model_savepath, "XGBoost-" + str(fold_counter))
            xgb_train_pass = XGBoost.train_model(
                withXGBoost[0], # n_estimators
                withXGBoost[1], # max_depth
                withXGBoost[2], # learning_rate
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                full_filename
            )
            xgboost_stats.append(xgb_train_pass)

        if withRandomForest:
            print("Training Random Forest")
            full_filename = os.path.join(model_savepath, "RandomForest-" + str(fold_counter))
            random_forest_pass = RandomForest.train_model(
                withRandomForest[0], # n_estimators
                withRandomForest[1], # max_depth
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                full_filename
            )
            random_forest_stats.append(random_forest_pass)

        if withLSTM2:
            print("Training 2-Class LSTM")
            tensorboard_logir = os.path.join(result_path, 'LSTM2C', str(fold_counter))
            mode_savepath = os.path.join(model_savepath, 'LSTM2C-' + str(fold_counter))
            lstm2_pass = LSTM_2_Multiclass.train_model(
                withLSTM2[0], # batch_size
                withLSTM2[1], # epochs
                withLSTM2[2], # learning_rate
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                tensorboard_logir,
                mode_savepath
            )
            lstm2_stats.append(lstm2_pass)

        if withHBOS:
            print("Training HBOS")
            full_filename = os.path.join(model_savepath, "HBOS")
            hbos_pass = HBOS.train_model(
                withHBOS[0], # n_bins
                withHBOS[1], # alpha
                withHBOS[2], # tol
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                full_filename
            )
            hbos_stats.append(hbos_pass)

        if withAutoEnc:
            print("Training AutoEncoder")
            full_filename = os.path.join(model_savepath, "AutoEncoder")
            auoenc_pass = Autoencoder.train_model(
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                full_filename,
                batch_size=withAutoEnc[0],    # batch
                epochs=withAutoEnc[1],        # epochs
                dropout_rate=withAutoEnc[2],  # dropout_rate
                contamination=withAutoEnc[3], # contamination
            )
            autoenc_stats.append(auoenc_pass)

        if withIForest:
            print("Training Isolation Forest")
            full_filename = os.path.join(model_savepath, "IsolationForest")
            iforest_pass = IsolationForest.train_model(
                fold_train_data,
                fold_train_labels,
                fold_test_data,
                fold_test_labels,
                full_filename,
                n_estimators=withIForest[0],
                contamination=withIForest[1],
                max_features=withIForest[2],
                bootstrap=withIForest[3]
            )
            iforest_stats.append(iforest_pass)

        fold_counter += 1

    # write fold-indices to disk
    filepath = os.path.join(result_path, 'dataset_fold_indices.json')
    with open(filepath, 'w') as mj:
        json.dump(fold_indices, mj)

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

    try:
        if withAutoEnc:
            metrics.calc_metrics_and_save_to_disk(autoenc_stats, 'AutoEncoder', result_path)
    except Exception:
        print('Ran into exception while saving AutoEncoder results to disk')

    try:
        if withIForest:
            metrics.calc_metrics_and_save_to_disk(iforest_stats, 'IsolationForest', result_path)
    except Exception:
        print('Ran into exception while saving IsolationForest results to disk')

    if args.ResultMail:
        result_mail.prepare_and_send_results(result_path, args)

    full_respath = os.path.abspath(result_path)
    print('Results have ben saved to {}'.format(full_respath))

    # Return the full path for other runners to use
    return full_respath
