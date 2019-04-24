import os
import sys

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import decomposition
import pandas as pd

sys.path.insert(0, os.path.abspath('./MLT'))
from MLT.tools import dataset_tools

CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'MLT', 'datasets', 'CICIDS2017pub'))

def cic_select_features():


    traind, trainl, testd, testl = 'cic_train_data_rand', 'cic_train_labels_rand', 'cic_test_data_rand', 'cic_test_labels_rand'
    cic_train_data = dataset_tools.load_df(traind, CIC_FOLDER_PATH)
    cic_train_labels = dataset_tools.load_df(trainl, CIC_FOLDER_PATH)
    cic_test_data = dataset_tools.load_df(testd, CIC_FOLDER_PATH)
    cic_test_labels = dataset_tools.load_df(testl, CIC_FOLDER_PATH)

    # concat all the files together
    cic_data = cic_test_data.append(cic_train_data, ignore_index=True, sort=False)
    cic_labels = cic_test_labels.append(cic_train_labels, ignore_index=True, sort=False)


    pd.options.display.max_rows = 500
    pd.options.display.max_columns = 500

    print(cic_data.columns.values)
    print("Column count original: {}".format(len(cic_data.columns.values)))

    pca = decomposition.PCA(n_components=16)
    pca.fit(cic_data)
    cic_16_test_data = pca.transform(cic_data)

    print(cic_16_test_data.head())
    print("Column count new: {}".format(len(cic_16_test_data.columns.values)))


def write_to_pickle(dataframe, filename):
    dataframe.to_pickle(os.path.join(CIC_FOLDER_PATH, filename+'.pkl'))

# This yielded [7, 14, 16, 20, 41, 42, 43, 44, 54, 56, 67, 68, 69, 80, 81, 82] on combined data:
# ['total_backward_packets', 'bwd_packet_length_max', 'bwd_packet_length_mean', 'flow_iat_mean',
# 'bwd_packets_per_s', 'min_packet_length', 'max_packet_length', 'packet_length_mean',
# 'ece_flag_count','average_packet_size', 'subflow_fwd_bytes', 'subflow_bwd_packets',
# 'subflow_bwd_bytes', 'idle_max', 'idle_min', 'source_ip_o1']

if __name__ == "__main__":
    cic_select_features()
