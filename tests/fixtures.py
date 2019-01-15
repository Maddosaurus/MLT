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
    dpath = tmpdir_factory.mktemp('data')
    with open(os.path.join(dpath, 'kdd_label_wordindex.json'), 'w') as outfile:
        json.dump(nsl_label_index, outfile)
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
