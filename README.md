<p align="center">
    <a href="https://github.com/Maddosaurus/MLT">
        <img src="https://img.shields.io/badge/status-1.0--prerelease-red.svg" alt="Status">
    </a>
    <a href="https://app.snyk.io/test/github/Maddosaurus/MLT?targetFile=rtd-requirements.txt">
        <img src="https://snyk.io/test/github/Maddosaurus/MLT/badge.svg?targetFile=rtd-requirements.txt" alt="Known Vulnerabilities">
    </a>
</p>

<p align="center">
    <a href='https://mlt.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/mlt/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://github.com/Maddosaurus/MLT/graphs/commit-activity">
        <img src="https://img.shields.io/badge/maintained-yes-brightgreen.svg" alt="Maintenance">
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

## Documentation
The current documentation can be found at [readthedocs.io](https://mlt.readthedocs.io/en/latest/).  
If you're intersted in manually building the API documentation, run `make html` in the `docroot` folder. This command will generate the full sphinx-doc for the project.
You can view a local copy of the docs by running `cd docroot/_build/html && python -m http.server` from the project root.

## Requirements
- Python 3.6+
- CUDA 9.1 (optional)

If you plan on using GPU-accelerated learning (strongly recommended), please set up CUDA 9.1 on your system. The current version of Tensorflow relies on CUDA 9.1 (not 10!). Please refer to the [Tensorflow Install How To](https://www.tensorflow.org/install/gpu) for up to date install instructions!  
If you are not using the deep learning potion, make sure to replace `tensorflow-gpu` with `tensorflow` in the requirements file.
The use of a virtual environment is strongly advised!  
All package requirements can be installed via `pipenv install` (add `--dev` for development dependencies).

Besides these, you will need copies of the *NSL-KDD* and *CICIDS2017* datasets stored in the subfolder `datasets`. The CICIDS2017 dataset is provided by the [University of New Brunswick](http://www.unb.ca/cic/datasets/index.html), while NSL-KDD can be obtained [freely on GitHub](https://github.com/defcom17/NSL_KDD). Additional datasets can be included analogous to these.  

## Workflow
The general workflow is:  
1. Dataset Preparation (sanitize and pickle)
2. Algorithm definition
3. Feature Selection, optional CV spits and Normalization/Scaling
4. Algorithm Training
5. Result Collection and Evaluation

## Dataset Preparation
I've sticked to the scheme of doing any sanitazation in `sanitize$DATASET.py`,  
while serialization to pickle as well as string encoding and misc stuff happens in `pickle$DATASET.py`.  
These steps are then imported to `MLT/run.py` and can be called through switches.  
