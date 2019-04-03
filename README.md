<p align="center">
    <a href='https://mlt.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/mlt/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://travis-ci.com/Maddosaurus/MLT'>
        <img src='https://img.shields.io/travis/com/Maddosaurus/MLT.svg' alt='TravisCI Status' />
    </a>
    <a href="https://codecov.io/gh/Maddosaurus/MLT">
        <img src="https://codecov.io/gh/Maddosaurus/MLT/branch/master/graph/badge.svg" alt='Code Coverage'/>
</a>
</p>

<p align="center">
    <a href="https://github.com/Maddosaurus/MLT/graphs/commit-activity">
        <img src="https://img.shields.io/badge/maintained-yes-brightgreen.svg" alt="Maintenance Status">
    </a>
    <a href="https://github.com/Maddosaurus/MLT/releases">
        <img src="https://img.shields.io/badge/version-1.0--prerelease-red.svg" alt="Version">
    </a>
    <a href="https://github.com/Maddosaurus/MLT/pulls">
        <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" alt="Contributions">
    </a>
</p>


# The MachineLearning Testbench
This piece of software prepares different datasets and ML algorithms as well as implementations and saves the qualitative benchmark results.  
It emerged as part of a CompSci Masters' Thesis at the University of Applied Sciences and Arts Dortmund.  

**It is currently in a state of prerelease and still subject to changes!**  
**A stable release can be expected around 03/2019**

## Getting Started
Have a look at the [Getting Started](https://mlt.readthedocs.io/en/latest/gettingstarted.html) section in the documentation for a detailed guide.  
Here is a minimal working example to check your installation:
```bash
git clone https://github.com/Maddosaurus/MLT
cd MLT
pipenv install
cd MLT/datasets
git clone https://github.com/defcom17/NSL_KDD NSL_KDD
cd ..
python run.py --pnsl
python run.py --single --nsl --xgb 10 10 0.1
```
Upon completion, you should be able to find infos for the test run in your console as well as in the subfolder `results`.

## Requirements
- Python 3.6+
- CUDA 9.1 (optional)
- tensorflow-gpu (optional)

If you plan on using GPU-accelerated learning (strongly recommended), please set up CUDA 9.1 on your system. The current version of Tensorflow relies on CUDA 9.1 (not 10!). Please refer to the [Tensorflow Install How To](https://www.tensorflow.org/install/gpu) for up to date install instructions!  
If you are interested in using the GPU-accelerated deep learning potion, make sure to replace `tensorflow` with `tensorflow-gpu` in your installation.
The use of a virtual environment is strongly advised!  
All package requirements can be installed via `pipenv install` (add `--dev` for development dependencies).

Besides these, you will need copies of the *NSL-KDD* and *CICIDS2017* datasets stored in the subfolder `datasets` (`/NSL_KDD` and `/CICIDS2017pub`). The CICIDS2017 dataset can be downloaded at the [University of New Brunswick](http://www.unb.ca/cic/datasets/index.html), while NSL-KDD can be obtained [on GitHub](https://github.com/defcom17/NSL_KDD). Additional datasets can be included analogous to these.  

## Documentation
The current documentation can be found at [readthedocs.io](https://mlt.readthedocs.io/en/latest/).  
If you're intersted in manually building the API documentation, run `make html` in the `docroot` folder. This command will generate the full sphinx-doc for the project.
You can view a local copy of the docs by running `cd docroot/_build/html && python -m http.server` from the project root.


## Workflow
The general workflow is:  
1. Dataset Preparation (sanitize and pickle)
2. Algorithm definition
3. Feature Selection, optional CV spits and Normalization/Scaling
4. Algorithm Training
5. Result Collection and Evaluation
