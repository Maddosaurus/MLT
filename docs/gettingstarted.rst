Getting Started
===============
Let's fire up a minimal working example to verify that
the installation succeeded and you can hit the ground running.
Before trying these steps, make sure you've set up everything
according to the :ref:`mlt-requirements`.

Tensorflow-GPU
----------------
First of all, if you are using tensorflow-gpu,
you should verify that your installation has been successful::

    import tensorflow as tf

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

This should result in the constant being printed.
Besides that, the debug output should include something along these lines::

    2019-01-04 09:48:22.560571: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
    name: GeForce GTX 980 major: 5 minor: 2 memoryClockRate(GHz): 1.304
    2019-01-04 09:48:23.258135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3042 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980, pci bus id: 0000:01:00.0, compute capability: 5.2)


Minimal Working Example
------------------------
I recommend to set up the NSL_KDD dataset, as it is considerably
small and quick to download / prepare.
You will need `this version of NSL_KDD <https://github.com/defcom17/NSL_KDD>`_
in the ``MLT/MLT/datasets/NSL_KDD`` folder::

    git clone https://github.com/defcom17/NSL_KDD NSL_KDD

The next step is the preparation of the pickles MLT uses::

    python run.py --pnsl

This will result in the creation of ``kdd_train_data.pkl``
and ``kdd_test_data.pkl`` amongst other supplemental caching and index files.
From here on, you can call your implementations with the ``--nsl6`` flag to
run them on the NSL_KDD dataset. 
A quick first test is to run a single benchmark run on NSL_KDD with XGBoost::

    python run.py --nsl6 --single --xgb 10 10 0.1

Afterwards, you can find all stats for the test run either on the console
or serialised in the folder ``MLT/results/NSL_6class_fb/DATE_TIME``.

