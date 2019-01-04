import pytest
import numpy

from .context import metrics_base
from .context import prediction_entry as pe


@pytest.fixture
def predictions(scope="module"):
    preds = []
    preds.append(pe.PredictionEntry(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        0))
    preds.append(pe.PredictionEntry(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        0))
    preds.append(pe.PredictionEntry(
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        0))
    return preds

def test_acc(predictions):
    assert metrics_base.calc_acc(predictions)[0] == 1
    assert metrics_base.calc_acc(predictions)[1] == 0.8
    assert metrics_base.calc_acc(predictions)[2] == 0.5
