"""Load the custom Splunk dataset from the pickle"""
import os
from MLT.tools import dataset_tools

SPLUNK_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Splunk'))


def get_splunk_full():
    """Use the full dataset with all features"""
    return _load_splunk()

def get_splunk_full_random():
    return _load_splunk(rand=True)

def _load_splunk(rand=False):
    """Loads the dataset and filters for given column names

    Returns
    -------
        data : tuple
            A tuple containing the filtered train- and test-data and -labels
    """
    if rand:
        train_data = dataset_tools.load_df('splunk_train_data_rand', SPLUNK_FOLDER_PATH)
        test_data = dataset_tools.load_df('splunk_test_data_rand', SPLUNK_FOLDER_PATH)
        train_labels = dataset_tools.load_df('splunk_train_labels_rand', SPLUNK_FOLDER_PATH)
        test_labels = dataset_tools.load_df('splunk_test_labels_rand', SPLUNK_FOLDER_PATH)
    else:
        train_data = dataset_tools.load_df('splunk_train_data', SPLUNK_FOLDER_PATH)
        test_data = dataset_tools.load_df('splunk_test_data', SPLUNK_FOLDER_PATH)
        train_labels = dataset_tools.load_df('splunk_train_labels', SPLUNK_FOLDER_PATH)
        test_labels = dataset_tools.load_df('splunk_test_labels', SPLUNK_FOLDER_PATH)

    print("")
    print("No of train entries:\t", len(train_data))
    print("No of train labels:\t", len(train_labels))
    print("-----------")
    print("No of test entries:\t", len(test_data))
    print("No of test labels:\t", len(test_labels))

    return(train_data, test_data, train_labels['label_encoded'].values, test_labels['label_encoded'].values)
