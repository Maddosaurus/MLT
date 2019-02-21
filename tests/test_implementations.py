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
    hbo = mlt_hbos._create_model(n_bins=42.0, alpha=0.4, tol=0.5)
    assert isinstance(hbo, HBOS)
    assert hbo.n_bins == 42
    assert hbo.alpha == 0.4
    assert hbo.tol == 0.5

def test_autoencoder():
    ae = mlt_autoencoder._create_model(epochs=2.0, batch_size=4.0, contamination=0.4, dropout_rate=0.11, hidden_neurons=[2,1,1,2])
    assert isinstance(ae, AutoEncoder)
    assert ae.epochs == 2
    assert ae.batch_size == 4
    assert ae.contamination == 0.4
    assert ae.dropout_rate == 0.11

def test_isolationforest():
    iforest= mlt_iforest._create_model(n_estimators=99.0, contamination=0.3, max_features=0.4, bootstrap=1.0)
    assert isinstance(iforest, IForest)
    assert iforest.n_estimators == 99
    assert iforest.contamination == 0.3
    assert iforest.max_features == 0.4
    assert iforest.bootstrap == True

def test_lstm():
    lstm = LSTM_2_Multiclass._create_model(learning_rate=0.001)
    assert isinstance(lstm, Sequential)

def test_random_forest():
    rf = RandomForest._createModel(n_estimators=33.0, max_depth=2.0)
    assert isinstance(rf, RandomForestClassifier)
    assert rf.n_estimators == 33
    assert rf.max_depth == 2

def test_xgboost():
    xgb = XGBoost._create_model(n_estimators=42.0, max_depth=1.0, learning_rate=0.3)
    assert isinstance(xgb, XGBClassifier)
    assert xgb.n_estimators == 42
    assert xgb.max_depth == 1
    assert xgb.learning_rate == 0.3