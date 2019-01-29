# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
import os
import json
from numpy.testing import assert_array_equal

from .fixtures import target_dir, load_mock_nsl_data, load_mock_nsl_labels, load_mock_cic_data, load_mock_cic_labels
from .context import single_benchmark, dataset_tools, NSL, CIC


def test_nsl6_filter(monkeypatch, target_dir, load_mock_nsl_data, load_mock_nsl_labels):
    def ret_testdir(somepath):
        return target_dir

    def ret_df(df_name, df_path):
        if df_name.endswith('data'):
            return load_mock_nsl_data
        elif df_name.endswith('labels'):
            return load_mock_nsl_labels

    def mock_os_join(dirpath, filename):
        if 'json' in filename:
            return target_dir + '/kdd_label_wordindex.json'
        return target_dir


    monkeypatch.setattr(dataset_tools, 'load_df', ret_df)
    monkeypatch.setattr(os.path, 'dirname', ret_testdir)
    monkeypatch.setattr(os.path, 'join', mock_os_join)

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

    def mock_os_join(dirpath, filename):
        if 'json' in filename:
            return target_dir + '/kdd_label_wordindex.json'
        return target_dir


    monkeypatch.setattr(dataset_tools, 'load_df', ret_df)
    monkeypatch.setattr(os.path, 'dirname', ret_testdir)
    monkeypatch.setattr(os.path, 'join', mock_os_join)

    tr_data, te_data, tr_labels, te_labels = NSL.get_NSL_16class()
    wanted_fields = [
        'service', 'flag', 'dst_bytes', 'wrong_fragment', 'count',
        'serror_rate', 'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate'
    ]

    assert_array_equal(tr_data.columns.values, wanted_fields)


def test_cic6_random(monkeypatch, target_dir, load_mock_cic_data, load_mock_cic_labels):
    tr_data = call_cic_filter(
        CIC.get_CIC_6class_randomized,
        monkeypatch, target_dir,
        load_mock_cic_data, load_mock_cic_labels
    )
    wanted_fields = [
        'flow_duration', 'protocol',
        'total_fwd_packets', 'total_backward_packets',
        'flow_packets_per_s', 'destination_port'
    ]

    assert_array_equal(tr_data.columns.values, wanted_fields)


def test_cic6_strat(monkeypatch, target_dir, load_mock_cic_data, load_mock_cic_labels):
    tr_data = call_cic_filter(
        CIC.get_CIC_6class_stratified,
        monkeypatch, target_dir,
        load_mock_cic_data, load_mock_cic_labels
    )
    wanted_fields = [
        'flow_duration', 'protocol',
        'total_fwd_packets', 'total_backward_packets',
        'flow_packets_per_s', 'destination_port'
    ]

    assert_array_equal(tr_data.columns.values, wanted_fields)


def test_cic28(monkeypatch, target_dir, load_mock_cic_data, load_mock_cic_labels):
    tr_data = call_cic_filter(
        CIC.get_CIC_28class,
        monkeypatch, target_dir,
        load_mock_cic_data, load_mock_cic_labels
    )
    wanted_fields = [
        'source_port', 'destination_port', 'protocol', 'total_fwd_packets',
        'total_backward_packets', 'flow_packets_per_s',
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
        'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
        'down_per_up_ratio', 'average_packet_size',
        'source_ip_o1', 'source_ip_o2', 'source_ip_o3', 'source_ip_o4',
        'destination_ip_o1', 'destination_ip_o2', 'destination_ip_o3', 'destination_ip_o4',
        'external_ip_o1', 'external_ip_o2', 'external_ip_o3', 'external_ip_o4'
    ]

    assert_array_equal(tr_data.columns.values, wanted_fields)


# Heavy lifting function for CIC
def call_cic_filter(filter_function, monkeypatch, target_dir, load_mock_cic_data, load_mock_cic_labels):
    def ret_testdir(somepath):
        return target_dir

    def ret_df(df_name, df_path):
        if df_name.endswith('data_randomized') or df_name.endswith('data_stratified'):
            return load_mock_cic_data
        elif df_name.endswith('labels_randomized') or df_name.endswith('labels_stratified'):
            return load_mock_cic_labels

    def mock_os_join(dirpath, filename):
        if 'json' in filename:
            return target_dir + '/cic_label_wordindex.json'
        return target_dir

    monkeypatch.setattr(dataset_tools, 'load_df', ret_df)
    monkeypatch.setattr(os.path, 'dirname', ret_testdir)
    monkeypatch.setattr(os.path, 'join', mock_os_join)

    tr_data, te_data, tr_labels, te_labels = filter_function()

    # for now, data suffices
    return tr_data
