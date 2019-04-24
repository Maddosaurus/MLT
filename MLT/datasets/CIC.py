"""Load the CICIDS2017 dataset from the pickle and filter features"""
import os
import numpy as np
from MLT.tools import dataset_tools

# Where to load the dataset pickles from
CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CICIDS2017pub'))

def get_CIC_Top20():
    """Get the randomized Top 20 class subset identified by mutual_info_classif.

    To generate these fields, call cic_feature_selection.py
    """
    # These were generated / found by running cic_feature_select
    # with Mutual Info Classif
    fields = [0, 3, 4, 5, 9, 11, 12, 17, 19, 22, 36, 37, 38, 39, 49, 51, 54, 56, 57, 58]
    return _load_cic(fields)

def get_CIC(transformed=False):
    return _load_cic(transformed=transformed)


def _load_cic(columns=None, transformed=False):
    """Load an return the feature dataset as tuple.

    Args:
        columns (list[int] or lsit[string], optional): List of columns to keep from the full dataset
        transformed (bool, optional): Whether to use a PowerTransformed version of the dataset

    Returns:
        data (tuple): A tuple containing train- and test-data and -labels
    """
    ## Data loading and prep
    if transformed:
        traind, trainl, testd, testl = 'cic_train_data_rand_yj', 'cic_train_labels_rand_yj', 'cic_test_data_rand_yj', 'cic_test_labels_rand_yj'
    else:
        traind, trainl, testd, testl = 'cic_train_data_rand', 'cic_train_labels_rand', 'cic_test_data_rand', 'cic_test_labels_rand'
    # As we've pickled the encoded dataset,
    # we only need to load these pickles to get the Pandas DataFrames back.
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    if columns is not None:
        if isinstance(columns[0], int):
            cic_train_data = cic_train_data.iloc[:, list(columns)]
            cic_test_data = cic_test_data.iloc[:, list(columns)]
        elif isinstance(columns[0], str):
            cic_train_data = cic_train_data.filter(columns)
            cic_test_data = cic_test_data.filter(columns)

    # ### Label translation
    # As we are doing binary classification,
    # we only need to know if the entry is normal/benign (*0*) or malicious (*1*)
    # Also, by default the encoding starts with BENIGN -> 1
    def translate_to_binary(label_value):
        return 0 if label_value == 1 else 1
    translate_to_binary = np.vectorize(translate_to_binary)

    cic_train_labels = translate_to_binary(cic_train_labels['label_encoded'].values)
    cic_test_labels = translate_to_binary(cic_test_labels['label_encoded'].values)
    print("")
    print("No of train entries:\t", len(cic_train_data))
    print("No of train labels:\t", len(cic_train_labels))
    print("-----------")
    print("No of test entries:\t", len(cic_test_data))
    print("No of test labels:\t", len(cic_test_labels))

    return (cic_train_data, cic_test_data, cic_train_labels, cic_test_labels)
