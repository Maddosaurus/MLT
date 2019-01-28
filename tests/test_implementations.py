# pylint: disable=redefined-outer-name,missing-docstring,unused-import,no-self-use
import pytest
from pyod.models.hbos import HBOS
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from .context import mlt_hbos, mlt_autoencoder, mlt_iforest, LSTM_2_Multiclass, RandomForest, XGBoost


def test_hbos():
    assert isinstance(mlt_hbos._create_model(), HBOS)

def test_autoencoder():
    assert isinstance(mlt_autoencoder._create_model(hidden_neurons=[2,1,1,2]), AutoEncoder)

def test_isolationforest():
    assert isinstance(mlt_iforest._create_model(), IForest)

def test_lstm():
    assert isinstance(LSTM_2_Multiclass._create_model(), Sequential)

def test_random_forest():
    assert isinstance(RandomForest._createModel(), RandomForestClassifier)

def test_xgboost():
    assert isinstance(XGBoost._create_model(), XGBClassifier)