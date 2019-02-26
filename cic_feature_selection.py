import os
import sys

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import pandas as pd

sys.path.insert(0, os.path.abspath('./MLT'))
from MLT.tools import dataset_tools

CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'CICIDS2017'))

def cic_select_features_f_classif():


    traind, trainl, testd, testl = 'cic_train_data_randomized', 'cic_train_labels_randomized', 'cic_test_data_randomized', 'cic_test_labels_randomized'
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    # concat all the files together
    cic_data = cic_test_data.append(cic_train_data, ignore_index=True, sort=False)
    cic_labels = cic_test_labels.append(cic_train_labels, ignore_index=True, sort=False)


    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500

    # drop label, flow_id and timestamp, as they break the selection mechanism and won't be needed
    cic_data.drop(['flow_id', 'timestamp'], axis=1, inplace=True)
    cic_labels.drop(['label'], axis=1, inplace=True)

    print(cic_data.head())
    print("Column count original: {}".format(len(cic_data.columns.values)))

    selector = SelectKBest(f_classif, k=16)
    selector.fit(cic_data, cic_labels.label_encoded.values.ravel())

    columns_int = selector.get_support(indices=True)
    with open(os.path.join(CIC_FOLDER_PATH, 'cic_top16_indices_fclassif.list'), 'w') as handle:
        handle.write(", ".join(str(x) for x in columns_int))

    columns = selector.get_support()
    cic_16_test_data = cic_test_data.iloc[:, list(columns)]
    print(cic_16_test_data.head())
    print("Column count new: {}".format(len(cic_16_test_data.columns.values)))

    # This yielded [12, 14, 15, 20, 24, 25, 41, 56, 76, 78, 79, 81, 88, 89, 90, 91] on combined data
    # fwd_packet_length_mean, bwd_packet_length_max, bwd_packet_length_min, flow_iat_mean, fwd_iat_total,
    # fwd_iat_mean, bwd_packets_per_s, average_packet_size, active_max, idle_mean, idle_std, idle_min,
    # destination_ip_o3, destination_ip_o4, external_ip_o1, external_ip_o2



def cic_select_features():


    traind, trainl, testd, testl = 'cic_train_data_randomized', 'cic_train_labels_randomized', 'cic_test_data_randomized', 'cic_test_labels_randomized'
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    # concat all the files together
    cic_data = cic_test_data.append(cic_train_data, ignore_index=True, sort=False)
    cic_labels = cic_test_labels.append(cic_train_labels, ignore_index=True, sort=False)


    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500

    # drop label, flow_id and timestamp, as they break the selection mechanism and won't be needed
    cic_data.drop(['flow_id', 'timestamp'], axis=1, inplace=True)
    cic_labels.drop(['label'], axis=1, inplace=True)

    print(cic_data.head())
    print("Column count original: {}".format(len(cic_data.columns.values)))

    selector = SelectKBest(mutual_info_classif, k=16)
    selector.fit(cic_data, cic_labels.label_encoded.values.ravel())

    columns_int = selector.get_support(indices=True)
    with open(os.path.join(CIC_FOLDER_PATH, 'cic_top16_indices.list'), 'w') as handle:
        handle.write(", ".join(str(x) for x in columns_int))

    columns = selector.get_support()
    cic_16_test_data = cic_test_data.iloc[:, list(columns)]
    print(cic_16_test_data.head())
    print("Column count new: {}".format(len(cic_16_test_data.columns.values)))


# This yielded [7, 14, 16, 20, 41, 42, 43, 44, 54, 56, 67, 68, 69, 80, 81, 82] on combined data:
# ['total_backward_packets', 'bwd_packet_length_max', 'bwd_packet_length_mean', 'flow_iat_mean',
# 'bwd_packets_per_s', 'min_packet_length', 'max_packet_length', 'packet_length_mean',
# 'ece_flag_count','average_packet_size', 'subflow_fwd_bytes', 'subflow_bwd_packets',
# 'subflow_bwd_bytes', 'idle_max', 'idle_min', 'source_ip_o1']

if __name__ == "__main__":
    print("F Classif with ANOVA-F:")
    cic_select_features_f_classif()
    print("\n\nMutual Info Classif:")
    cic_select_features()
