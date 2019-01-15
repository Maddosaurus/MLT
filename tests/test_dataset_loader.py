# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
import os
from numpy.testing import assert_array_equal

from .fixtures import target_dir, load_mock_nsl_data, load_mock_nsl_labels
from .context import single_benchmark, dataset_tools, NSL


def test_nsl6_filter(monkeypatch, target_dir, load_mock_nsl_data, load_mock_nsl_labels):
    def ret_testdir(somepath):
        return target_dir

    def ret_df(df_name, df_path):
        if df_name.endswith('data'):
            return load_mock_nsl_data
        elif df_name.endswith('labels'):
            return load_mock_nsl_labels

    monkeypatch.setattr(dataset_tools, 'load_df', ret_df)
    monkeypatch.setattr(os.path, 'dirname', ret_testdir)

    tr_data, te_data, tr_labels, te_labels = NSL.get_NSL_6class()
    wanted_fields = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'count', 'srv_count']

    assert_array_equal(tr_data.columns.values, wanted_fields)


def test_nsl16_filter(monkeypatch, target_dir, load_mock_nsl_data, load_mock_nsl_labels):
    def ret_testdir(somepath):
        return target_dir

    def ret_df(df_name, df_path):
        if df_name.endswith('data'):
            return load_mock_nsl_data
        elif df_name.endswith('labels'):
            return load_mock_nsl_labels

    monkeypatch.setattr(dataset_tools, 'load_df', ret_df)
    monkeypatch.setattr(os.path, 'dirname', ret_testdir)

    tr_data, te_data, tr_labels, te_labels = NSL.get_NSL_16class()
    wanted_fields = [
        'service', 'flag', 'dst_bytes', 'wrong_fragment', 'count',
        'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate'
    ]

    assert_array_equal(tr_data.columns.values, wanted_fields)
