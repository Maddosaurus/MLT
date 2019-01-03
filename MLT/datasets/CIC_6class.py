"""Load the CICIDS2017 dataset from the pickle and choose 6 attributes similar to NSL_KDD"""
import os
import json
import numpy as np
from MLT.tools import dataset_tools

# Where to load the dataset pickles from
CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), '..', 'datasets', 'CICIDS2017'))

def get_CIC_6class(stratified=True):
    """Load an return the stratified 6 feature dataset as tuple"""
    ## Data loading and prep
    if stratified:
        traind, trainl, testd, testl = 'cic_train_data_stratified','cic_train_labels_stratified','cic_test_data_stratified','cic_test_labels_stratified'
    else:
        traind, trainl, testd, testl = 'cic_train_data_randomized','cic_train_labels_randomized','cic_test_data_randomized','cic_test_labels_randomized'

    # As we've pickled the encoded dataset,
    # we only need to load these pickles to get the Pandas DataFrames back.
    # **Hint**: If you miss the pickles, go ahead and run *main.py --pickeCIC*
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    # We are only using 6 features that are somewhat similar to NSL_KDD
    used_fields = [
        'flow_duration', 'protocol',
        'total_fwd_packets', 'total_backward_packets',
        'flow_packets_per_s', 'destination_port'
    ]
    cic_train_data = cic_train_data.filter(used_fields)
    cic_test_data = cic_test_data.filter(used_fields)

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


def get_CIC_6class_stratified():
    """Load the stratified 6 class subset of CICIDS2017."""
    return get_CIC_6class(stratified=True)

def get_CIC_6class_randomized():
    """Load the randomized 6 class subset of CICIDS2017."""
    return get_CIC_6class(stratified=False)