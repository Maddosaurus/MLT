"""Main context file that fixed the import / packet distribution dilemma.

See https://docs.python-guide.org/writing/structure/#test-suite on the how and why."""

#pylint: disable=unused-import,wrong-import-position

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../MLT')))

import MLT
from MLT.implementations import HBOS as mlt_hbos
from MLT.implementations import LSTM_2_Multiclass, RandomForest, XGBoost

from MLT.metrics import metrics_base, metrics_cm
from MLT.tools import prediction_entry
