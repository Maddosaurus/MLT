# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
from datetime import timedelta
import pytest
import numpy as np
import os
import json

from .fixtures import predictions, target_dir
from .context import metrics_base, metrics_cm

class TestMetricBase:
    def test_acc(self, predictions):
        assert metrics_base.calc_acc(predictions)[0] == 1
        assert metrics_base.calc_acc(predictions)[1] == 0.6
        assert metrics_base.calc_acc(predictions)[2] == 0.5

    def test_precision(self, predictions):
        assert metrics_base.calc_precision(predictions)[0] == 1
        assert metrics_base.calc_precision(predictions)[1] == 1
        assert metrics_base.calc_precision(predictions)[2] == 0.5

    def test_recall(self, predictions):
        assert metrics_base.calc_recall(predictions)[0] == 1
        assert metrics_base.calc_recall(predictions)[1] == 0.6
        assert metrics_base.calc_recall(predictions)[2] == 1

    def test_f1(self, predictions):
        assert metrics_base.calc_fbeta_binary(predictions, 1)[0] == 1
        assert metrics_base.calc_fbeta_binary(predictions, 1)[1] == pytest.approx(0.75)
        assert metrics_base.calc_fbeta_binary(predictions, 1)[2] == pytest.approx(0.667, abs=1e-3)

    def test_sum_training_times(self, predictions):
        assert metrics_base.sum_training_times(predictions) == str(timedelta(4))

    def test_mean_training_time(self, predictions):
        assert metrics_base.calc_mean_training_time(predictions) == str(timedelta(1))


class TestMetricCM:
    def test_calc_cm(self, predictions):
        result = [[1, 3], [2, 4]] # [[TN, FP], [FN, TP]]
        cms = metrics_cm.calc_cm(predictions)
        np.testing.assert_array_equal(cms[3], result)

    def test_cm_to_disk(self, predictions, target_dir):
        result = [1, 3, 2, 4]
        
        cms = metrics_cm.calc_cm(predictions)
        metrics_cm.save_cm_arr_to_disk(cms, "pytest", target_dir)
        
        json_path = os.path.join(target_dir, "pytest_cms.json")
        with open(json_path) as jcms:
            loaded = json.load(jcms)["absolute"]["fold4"]
            np.testing.assert_array_equal(loaded, result)
