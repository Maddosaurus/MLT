MLT Modules
===============
These are the main modules of MLT.
Their usage and details are explained in detail for every module.


.. _mlt-datasets:

MLT.datasets
-------------------
Module for preparing individual datasets.

.. automodule:: MLT.datasets
    :members:

CICIDS2017
^^^^^^^^^^^^
.. automodule:: MLT.datasets.CIC_6class
    :members:
.. automodule:: MLT.datasets.pickleCIC
    :members:
.. automodule:: MLT.datasets.pickleNSL
    :members:

NSL_KDD
^^^^^^^^^
.. automodule:: MLT.datasets.NSL_6class
    :members:
.. automodule:: MLT.datasets.sanitizeCIC
    :members:

.. _mlt-implementations:

MLT.implementations
-------------------
This module contains the specific implementations to benchmark

.. automodule:: MLT.implementations
    :members:

XGBoost
^^^^^^^^^^^^^
.. automodule:: MLT.implementations.XGBoost
    :members:

RandomForest
^^^^^^^^^^^^^
.. automodule:: MLT.implementations.RandomForest
    :members:

HBOS
^^^^^^^^^^^^^
.. automodule:: MLT.implementations.HBOS
    :members:

LSTM_2_Multiclass
^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.implementations.LSTM_2_Multiclass
    :members:

.. _mlt-metrics:

MLT.metrics
-------------------
Generates advanced metrics for results and datasets.

.. automodule:: MLT.metrics
    :members:

Base Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.metrics.metrics_base
    :members:

Metrics related to Confusion Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.metrics.metrics_cm
    :members:

Feature Distribution Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.metrics.metrics_distrib
    :members:

ROC and AUC Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.metrics.metrics_roc
    :members:


.. _mlt-testrunners:

MLT.testrunners
-------------------
These runners are responsible for the benchmark execution and additional features like crossvalidation.

.. automodule:: MLT.testrunners
    :members:

Benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.testrunners.single_benchmark
    :members:

K-Fold Crossvalidation
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.testrunners.kfold_runner
    :members:


.. _mlt-tools:

MLT.tools
-------------------
A collection of misc tools that support the main modules.

.. automodule:: MLT.tools
    :members:

PredictionEntry
^^^^^^^^^^^^^^^^
.. automodule:: MLT.tools.prediction_entry
    :members:

Dataset Tools
^^^^^^^^^^^^^
.. automodule:: MLT.tools.dataset_tools
    :members:

Keras Helper
^^^^^^^^^^^^^
.. automodule:: MLT.tools.helper_keras
    :members:

Pyod Helper
^^^^^^^^^^^^^
.. automodule:: MLT.tools.helper_pyod
    :members:

Scikit Helper
^^^^^^^^^^^^^
.. automodule:: MLT.tools.helper_sklearn
    :members:

Email Tools
^^^^^^^^^^^^^
.. automodule:: MLT.tools.result_mail
    :members:

Result Helper
^^^^^^^^^^^^^^^
.. automodule:: MLT.tools.result_helper
    :members:

Uncategorized Tools
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: MLT.tools.toolbelt
    :members:
