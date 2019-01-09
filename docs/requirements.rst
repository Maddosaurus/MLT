.. _mlt-requirements:

Requirements
===============

This framework is actively developed on Windows and Linux,
OSX is tested sometimes, so it should mostly be fine.
A Pipfile.lock is not tracked by the VCS because some stuff breaks
between OSes, mostly regarding Tensorflow and CUDA.

Most of the requirements are covered in the Pipfile,
but there are some things to consider:

* Python 3.6+
* tensorflow-gpu (optional)
* CUDA 9.1 (optional)
* cuDNN (optional)

CUDA and cuDNN are only needed, if you plan on using tensorflow-gpu.
If you plan on using the deep networks, I would strongly recommend to set up tensorflow-gpu, which in turn needs CUDA 9.1 and cuDNN. Please refer to the up-to-date `install guide for tensorflow-gpu <https://www.tensorflow.org/install/gpu>`_.
Afterwards, make sure to remove the base tensorflow module and replace it with tensorflow-gpu::

    pipenv uninstall tensorflow --skip-lock
    pipenv install tensorflow-gpu


.. warning::
    Please note that at the time of writing CUDA 9.1 is needed, insted of the latest CUDA 10, as tensorflow 1.12 depends on this!
