Design Principles
=================
The general layout is built to be as modular as possible.  
Therefore, any module consists of multiple submodules that implement the main functionality.  

The runners in :ref:`mlt-testrunners` run the selected :ref:`mlt-implementations`  
on the given :ref:`mlt-datasets` and generate :ref:`mlt-metrics` after the run.  

All outcomes and collected data is then stored in the ``results`` folder,  
more specifically in ``results/$dataset_name/$isodate``.