# pylint: disable=missing-docstring,unused-argument
from datetime import timedelta
import pytest
import os
import json
import numpy as np
import pandas as pd

from .context import prediction_entry as pe


@pytest.fixture(scope="session")
def predictions():
    preds = []
    preds.append(pe.PredictionEntry(
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        timedelta(1)))
    preds.append(pe.PredictionEntry(
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        timedelta(2)))
    preds.append(pe.PredictionEntry(
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        timedelta(1)))
    preds.append(pe.PredictionEntry(
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        timedelta(0)))
    return preds


@pytest.fixture(scope='session')
def target_dir(tmpdir_factory):
    nsl_label_index = {"normal": 1, "neptune": 2, "warezclient": 3, "ipsweep": 4, "portsweep": 5, "teardrop": 6, "nmap": 7, "satan": 8, "smurf": 9, "pod": 10, "back": 11, "guess_passwd": 12, "ftp_write": 13, "multihop": 14, "rootkit": 15, "buffer_overflow": 16, "imap": 17, "warezmaster": 18, "phf": 19, "land": 20, "loadmodule": 21, "spy": 22, "perl": 23, "saint": 24, "mscan": 25, "apache2": 26, "snmpgetattack": 27, "processtable": 28, "httptunnel": 29, "ps": 30, "snmpguess": 31, "mailbomb": 32, "named": 33, "sendmail": 34, "xterm": 35, "worm": 36, "xlock": 37, "xsnoop": 38, "sqlattack": 39, "udpstorm": 40}
    cic_label_index = {"benign": 1, "ftppatator": 2, "sshpatator": 3, "dosslowloris": 4, "dosslowhttptest": 5, "doshulk": 6, "dosgoldeneye": 7, "heartbleed": 8, "bruteforce": 9, "xss": 10, "sqlinjection": 11, "infiltration": 12, "bot": 13, "portscan": 14, "ddos": 15}
    cic_top16_index = [7, 14, 16, 20, 41, 42, 43, 44, 54, 56, 67, 68, 69, 80, 81, 82]
    dpath = tmpdir_factory.mktemp('data')
    with open(os.path.join(dpath, 'kdd_label_wordindex.json'), 'w') as outfile:
        json.dump(nsl_label_index, outfile)
    with open(os.path.join(dpath, 'cic_label_wordindex.json'), 'w') as outfile:
        json.dump(cic_label_index, outfile)
    with open(os.path.join(dpath, 'cic_top16_indices.list'), 'w') as handle:
        handle.write(", ".join(str(x) for x in cic_top16_index))
    return dpath


@pytest.fixture(scope='session')
def load_mock_nsl_data():
    data_header = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'protocol_type', 'service', 'flag'
    ]
    mock_data = pd.DataFrame(data=np.ones((1, 41)), columns=data_header)
    return mock_data


@pytest.fixture(scope='session')
def load_mock_nsl_labels():
    mock_labels = pd.DataFrame(data={'label': ['neptune'], 'difficulty_level': [1], 'label_encoded': [1]})
    return mock_labels


@pytest.fixture(scope='session')
def load_mock_cic_data():
    data_header = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
        'Destination Port_-2', 'Destination Port_-1', 'Destination Port_0', 'Destination Port_1', 'Destination Port_3', 'Destination Port_4',
        'Destination Port_6', 'Destination Port_7', 'Destination Port_9', 'Destination Port_13', 'Destination Port_17', 'Destination Port_19',
        'Destination Port_20', 'Destination Port_21', 'Destination Port_22', 'Destination Port_23', 'Destination Port_24',
        'Destination Port_25', 'Destination Port_26', 'Destination Port_30', 'Destination Port_32', 'Destination Port_33',
        'Destination Port_37', 'Destination Port_42', 'Destination Port_43', 'Destination Port_49', 'Destination Port_53',
        'Destination Port_70', 'Destination Port_79', 'Destination Port_80', 'Destination Port_81', 'Destination Port_82',
        'Destination Port_83', 'Destination Port_84', 'Destination Port_85', 'Destination Port_88', 'Destination Port_89',
        'Destination Port_90', 'Destination Port_99', 'Destination Port_100', 'Destination Port_106', 'Destination Port_109',
        'Destination Port_110', 'Destination Port_111', 'Destination Port_113', 'Destination Port_119', 'Destination Port_123',
        'Destination Port_125', 'Destination Port_135', 'Destination Port_137', 'Destination Port_138', 'Destination Port_139',
        'Destination Port_143', 'Destination Port_144', 'Destination Port_146', 'Destination Port_161', 'Destination Port_163',
        'Destination Port_179', 'Destination Port_199', 'Destination Port_211', 'Destination Port_212', 'Destination Port_222',
        'Destination Port_254', 'Destination Port_255', 'Destination Port_256', 'Destination Port_259', 'Destination Port_264',
        'Destination Port_280', 'Destination Port_301', 'Destination Port_306', 'Destination Port_311', 'Destination Port_340',
        'Destination Port_366', 'Destination Port_389', 'Destination Port_406', 'Destination Port_407', 'Destination Port_416',
        'Destination Port_417', 'Destination Port_425', 'Destination Port_427', 'Destination Port_443', 'Destination Port_444',
        'Destination Port_445', 'Destination Port_458', 'Destination Port_464', 'Destination Port_465', 'Destination Port_481',
        'Destination Port_497', 'Destination Port_500', 'Destination Port_512', 'Destination Port_513', 'Destination Port_514',
        'Destination Port_515', 'Destination Port_524', 'Destination Port_541', 'Destination Port_543', 'Destination Port_544',
        'Destination Port_545', 'Destination Port_548', 'Destination Port_554', 'Destination Port_555', 'Destination Port_563',
        'Destination Port_587', 'Destination Port_593', 'Destination Port_616', 'Destination Port_617', 'Destination Port_625',
        'Destination Port_631', 'Destination Port_636', 'Destination Port_646', 'Destination Port_648', 'Destination Port_666',
        'Destination Port_667', 'Destination Port_668', 'Destination Port_683', 'Destination Port_687', 'Destination Port_691',
        'Destination Port_700', 'Destination Port_705', 'Destination Port_711', 'Destination Port_714', 'Destination Port_720',
        'Destination Port_722', 'Destination Port_726', 'Destination Port_749', 'Destination Port_765', 'Destination Port_777',
        'Destination Port_783', 'Destination Port_787', 'Destination Port_800', 'Destination Port_801', 'Destination Port_808',
        'Destination Port_843', 'Destination Port_873', 'Destination Port_880', 'Destination Port_888', 'Destination Port_898',
        'Destination Port_900', 'Destination Port_901', 'Destination Port_902', 'Destination Port_903', 'Destination Port_911',
        'Destination Port_912', 'Destination Port_981', 'Destination Port_987', 'Destination Port_990', 'Destination Port_992',
        'Destination Port_993', 'Destination Port_995', 'Destination Port_999', 'Destination Port_1000', 'Destination Port_1001',
        'Destination Port_1002', 'Destination Port_1007', 'Destination Port_1009', 'Destination Port_1010', 'Destination Port_1011',
        'Destination Port_1021', 'Destination Port_1022'
    ]
    mock_data = pd.DataFrame(data=np.ones((1, 228)), columns=data_header)
    return mock_data

@pytest.fixture(scope='session')
def load_mock_cic_labels():
    mock_labels = pd.DataFrame(data={'label': ['PortScan'], 'label_encoded': [14]})
    return mock_labels
