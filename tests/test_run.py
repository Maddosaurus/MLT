# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
import argparse

from .context import run, base_runner

def test_create_parser():
    assert isinstance(run.create_parser(), argparse.ArgumentParser)

def test_nsl_kfold(monkeypatch):
    def mock_run_nsl(args):
        assert args.kfolds == 2
        assert args.AutoEncoder[0] == 32.0
        assert args.AutoEncoder[1] == 100.0
        assert args.AutoEncoder[2] == 0.2
        assert args.AutoEncoder[3] == 0.1

    monkeypatch.setattr(base_runner, 'run_NSL', mock_run_nsl)
    parser = run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '2', '--nsl16', '--AutoEncoder', '32', '100', '0.2', '0.1'])
    run.main(args)

def test_nsl_single(monkeypatch):
    def mock_run_nsl(args):
        assert args.AutoEncoder[0] == 32.0
        assert args.AutoEncoder[1] == 100.0
        assert args.AutoEncoder[2] == 0.2
        assert args.AutoEncoder[3] == 0.1

    monkeypatch.setattr(base_runner, 'run_NSL', mock_run_nsl)
    parser = run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--nsl16', '--AutoEncoder', '32', '100', '0.2', '0.1'])
    run.main(args)


def test_cic_kfold(monkeypatch):
    def mock_run_cic(args):
        assert args.kfolds == 2
        assert args.AutoEncoder[0] == 32.0
        assert args.AutoEncoder[1] == 100.0
        assert args.AutoEncoder[2] == 0.2
        assert args.AutoEncoder[3] == 0.1

    monkeypatch.setattr(base_runner, 'run_CIC', mock_run_cic)
    parser = run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '2', '--cic20', '--AutoEncoder', '32', '100', '0.2', '0.1'])
    run.main(args)

def test_cic_single(monkeypatch):
    def mock_run_cic(args):
        assert args.AutoEncoder[0] == 32.0
        assert args.AutoEncoder[1] == 100.0
        assert args.AutoEncoder[2] == 0.2
        assert args.AutoEncoder[3] == 0.1

    monkeypatch.setattr(base_runner, 'run_CIC', mock_run_cic)
    parser = run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--cic20', '--AutoEncoder', '32', '100', '0.2', '0.1'])
    run.main(args)