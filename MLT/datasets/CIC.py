"""Load the CICIDS2017 dataset from the pickle and filter features"""
import os
import json
import numpy as np
from MLT.tools import dataset_tools

# Where to load the dataset pickles from
CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CICIDS2017'))


def get_CIC_6class_stratified():
    """Load the stratified 6 class subset of CICIDS2017."""
    fields = [
        'flow_duration', 'protocol',
        'total_fwd_packets', 'total_backward_packets',
        'flow_packets_per_s', 'destination_port'
    ]
    return _load_cic(fields, stratified=True)


def get_CIC_6class_randomized():
    """Load the randomized 6 class subset of CICIDS2017."""
    fields = [
        'flow_duration', 'protocol',
        'total_fwd_packets', 'total_backward_packets',
        'flow_packets_per_s', 'destination_port'
    ]
    return _load_cic(fields, stratified=False)


def get_CIC_28class():
    """Get the extended randomized 28 class subset of CICIDS2017"""
    fields = [
        'source_port', 'destination_port', 'protocol', 'total_fwd_packets',
        'total_backward_packets', 'flow_packets_per_s',
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
        'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
        'down_per_up_ratio', 'average_packet_size',
        'source_ip_o1', 'source_ip_o2', 'source_ip_o3', 'source_ip_o4',
        'destination_ip_o1', 'destination_ip_o2', 'destination_ip_o3', 'destination_ip_o4',
        'external_ip_o1', 'external_ip_o2', 'external_ip_o3', 'external_ip_o4'
    ]
    return _load_cic(fields, stratified=False)


def _load_cic(column_names, stratified=True):
    """Load an return the stratified 6 feature dataset as tuple

    Parameters
    ----------
        stratified : bool, optional (default=True)
            Whether to use stratified or randomized sampling

    Returns
    -------
        data : tuple
            A tuple containing the filtered train- and test-data and -labels
    """
    ## Data loading and prep
    if stratified:
        traind, trainl, testd, testl = 'cic_train_data_stratified', 'cic_train_labels_stratified', 'cic_test_data_stratified', 'cic_test_labels_stratified'
    else:
        traind, trainl, testd, testl = 'cic_train_data_randomized', 'cic_train_labels_randomized', 'cic_test_data_randomized', 'cic_test_labels_randomized'

    # As we've pickled the encoded dataset,
    # we only need to load these pickles to get the Pandas DataFrames back.
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    cic_train_data = cic_train_data.filter(column_names)
    cic_test_data = cic_test_data.filter(column_names)

    # ### Label translation
    # As we are doing binary classification,
    # we only need to know if the entry is normal/benign (*0*) or malicious (*1*)
    with open(os.path.join(CIC_FOLDER_PATH, 'cic_label_wordindex.json')) as json_in:
        data = json.load(json_in)
        print('Loaded these labels from Tokenization process:')
        print(data)
        normal_index = data['benign']

    def translate_to_binary(label_value):
        return 0 if label_value == normal_index else 1
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


