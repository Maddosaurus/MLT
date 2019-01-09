Tests
=======
To ensure the correct inner workings of the used frameworks and calculations,
the usage of unit and integration tests is advised.
Tests are executed by invoking ``pytest``, which will be installed
if you have installed the dev dependencies with ``pipenv install --dev``.
On branch push, `TravisCI`_ checks out all changes, calls ``pytest --cov=./``
and submits the coverage statistics to `codecov`_.

Test Structure
---------------
The main root for all tests is ``MLT/tests``.
It contains one file per logical module, accompanied by a ``context.py`` and ``fixtures.py``.
The former enables you to skip local package installation during development,
whilst the latter contains all custom fixtures used in tests.
If a logical module contains multiple submodules or complex code, test classes with subfunctions are used.

Adding new tests
-------------------
To add a new test:
    - import the subject in ``context.py``
    - add new ``test_new_module.py`` file
    - add tests in the file (use classes if necessary).


.. _TravisCI: https://travis-ci.com/Maddosaurus/MLT
.. _codecov: https://codecov.io/gh/Maddosaurus/MLT