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
    dpath = tmpdir_factory.mktemp('data')
    with open(os.path.join(dpath, 'kdd_label_wordindex.json'), 'w') as outfile:
        json.dump(nsl_label_index, outfile)
    with open(os.path.join(dpath, 'cic_label_wordindex.json'), 'w') as outfile:
        json.dump(cic_label_index, outfile)
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
        'flow_id', 'source_port', 'destination_port', 'protocol', 'timestamp',
        'flow_duration', 'total_fwd_packets', 'total_backward_packets',
        'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
        'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
        'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min',
        'bwd_packet_length_mean', 'bwd_packet_length_std', 'flow_bytes_per_s',
        'flow_packets_per_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
        'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
        'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
        'fwd_header_length', 'bwd_header_length', 'fwd_packets_per_s', 'bwd_packets_per_s',
        'min_packet_length', 'max_packet_length', 'packet_length_mean', 'packet_length_std',
        'packet_length_variance', 'fin_flag_count', 'syn_flag_count', 'rst_flag_count',
        'psh_flag_count', 'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
        'down_per_up_ratio', 'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size',
        'fwd_header_length.1', 'fwd_avg_bytes_per_bulk', 'fwd_avg_packets_per_bulk',
        'fwd_avg_bulk_rate', 'bwd_avg_bytes_per_bulk', 'bwd_avg_packets_per_bulk',
        'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets',
        'subflow_bwd_bytes', 'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean',
        'idle_std', 'idle_max', 'idle_min', 'source_ip_o1', 'source_ip_o2', 'source_ip_o3',
        'source_ip_o4', 'destination_ip_o1', 'destination_ip_o2', 'destination_ip_o3', 'destination_ip_o4',
        'external_ip_o1', 'external_ip_o2', 'external_ip_o3', 'external_ip_o4'
    ]
    mock_data = pd.DataFrame(data=np.ones((1, 94)), columns=data_header)
    return mock_data

@pytest.fixture(scope='session')
def load_mock_cic_labels():
    mock_labels = pd.DataFrame(data={'label': ['PortScan'], 'label_encoded': [14]})
    return mock_labels
