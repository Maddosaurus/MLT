# pylint: disable=missing-docstring,unused-argument
from datetime import timedelta
import pytest
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
    dpath = tmpdir_factory.mktemp('data')
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
