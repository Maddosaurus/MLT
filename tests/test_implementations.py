# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
from pyod.models.hbos import HBOS
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from .context import mlt_hbos, LSTM_2_Multiclass, RandomForest, XGBoost


def test_hbos():
    assert isinstance(mlt_hbos._create_model(), HBOS)

def test_lstm():
    assert isinstance(LSTM_2_Multiclass._create_model(), Sequential)

def test_random_forest():
    assert isinstance(RandomForest._createModel(), RandomForestClassifier)

def test_xgboost():
    assert isinstance(XGBoost._create_model(), XGBClassifier)