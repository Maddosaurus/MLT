# pylint: disable=missing-docstring,unused-argument
from datetime import timedelta
import pytest
import numpy as np

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
