# pylint: disable=C0103
"""Load the NSL_KDD dataset from the pickle and filter for attributes"""
import json
import os
import numpy as np
from MLT.tools import dataset_tools
# Where to load the NSL dataset from
NSL_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), '..', 'datasets', 'NSL_KDD'))


def get_NSL_6class():
    """Load the dataset, choose 6 features and binarize the labels"""
    used_fields = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'count', 'srv_count']
    return _load_nsl(used_fields)

def get_NSL_16class():
    """Load the dataset, choose 16 features based on Iglesias & Zseby (2015) and binarize labels"""
    used_fields = [
        'service', 'flag', 'dst_bytes', 'wrong_fragment', 'count',
        'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate'
    ]
    return _load_nsl(used_fields)



def _load_nsl(column_names):
    """Loads the dataset and filters for given column names

    args:
      column_names: String-Array of column names that you want in your dataset.

    returns:
      tuple containing (train_data, test_data, train_labels, test_labels)
    """
    # ## Data loading and prep

    # As we've pickled the encoded dataset,
    # we only need to load these pickles to get the Pandas DataFrames back.
    # **Hint**: If you miss the pickles, go ahead and run *pickle-NSL* or *main.py --pickeNSL*
    kdd_train_data = dataset_tools.load_df('kdd_train_data', NSL_FOLDER_PATH)
    kdd_test_data = dataset_tools.load_df('kdd_test_data', NSL_FOLDER_PATH)
    kdd_train_labels = dataset_tools.load_df('kdd_train_labels', NSL_FOLDER_PATH)
    kdd_test_labels = dataset_tools.load_df('kdd_test_labels', NSL_FOLDER_PATH)

    # The paper mentions that they only use six features of the full dataset
    # which is why we filter the dataframes for these.
    kdd_train_data = kdd_train_data.filter(column_names)
    kdd_test_data = kdd_test_data.filter(column_names)

    # ### Label translation
    # As we are doing binary classification,
    # we only need to know if the entry is normal/benign (*0*) or malicious (*1*)
    with open(os.path.join(NSL_FOLDER_PATH, 'kdd_label_wordindex.json')) as json_in:
        data = json.load(json_in)
        print('Loaded these labels from Tokenization process:')
        print(data)
        normal_index = data['normal']

    def translate_to_binary(label_value):
        return 0 if label_value == normal_index else 1
    translate_to_binary = np.vectorize(translate_to_binary)

    # We only want to know if it's benign or not, so we switch to 0 or 1
    kdd_train_labels = translate_to_binary(kdd_train_labels['label_encoded'].values)
    kdd_test_labels = translate_to_binary(kdd_test_labels['label_encoded'].values)

    print("")
    print("No of train entries:\t", len(kdd_train_data))
    print("No of train labels:\t", len(kdd_train_labels))
    print("-----------")
    print("No of test entries:\t", len(kdd_test_data))
    print("No of test labels:\t", len(kdd_test_labels))

    return (kdd_train_data, kdd_test_data, kdd_train_labels, kdd_test_labels)
