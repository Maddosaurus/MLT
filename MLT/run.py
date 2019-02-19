#! /usr/bin/env python
"""Main entrypoint and CLI interface for MLT"""

# pylint: disable=C0301,C0413

import warnings
import os
import sys
import argparse
from datetime import datetime


# Fixing broken MLT imports
sys.path.insert(0, os.path.abspath('..'))

# fix pyplot crashing on CLI-only systems.
# See https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# import the MLT-relevant parts
from MLT.datasets import pickleNSL
from MLT.datasets import pickleCIC
from MLT.datasets import sanitizeCIC
from MLT.metrics import metrics_roc
from MLT.testrunners import base_runner
from MLT.testrunners import kfold_runner
from MLT.testrunners import single_benchmark
from MLT.tools import result_helper



# Silence some warnings. See
# https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='numpy.dtype size changed*')
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning, message='The truth value of an empty array is ambiguous*')
warnings.filterwarnings(module='sklearn*', action='ignore', message='Data with input dtype int64 was converted to float64 by MinMaxScaler*')



def main(args=None):
    """Main CLI entrypoint"""
    starttime = datetime.now()

    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # enter main stuff
    print("\nG'day! Starting execution at {}".format(starttime))

    # tools
    if args.GenerateCVROCs:
        metrics_roc.generate_cv_roc_model_selection(
            args.GenerateCVROCs[0],
            args.GenerateCVROCs[1].split(','),
            args.GenerateCVROCs[2],
            args.GenerateCVROCs[3].split(','),
            args.GenerateCVROCs[4].split(','),
        )

    if args.ListResults:
        result_helper.list_scores(args.ListResults[0], args.ListResults[1])

    # dataset stuff
    if args.sanitizeCIC:
        print('Sanitizing CICIDS17')
        sanitizeCIC.sanitize_CICIDS2017()

    if args.pickleCICs:
        print('Stratified Pickle CICIDS2017!')
        pickleCIC.pickleCIC_stratified()

    if args.pickleCICr:
        print('Randomized Pickle CICIDS2017!')
        pickleCIC.pickleCIC_randomized()

    if args.pickleNSL:
        print('Pickle NSL_KDD!')
        pickleNSL.pickle_NSL_KDD()


    full_resultpath = ""

    if (args.NSL6 or args.NSL16):
        full_resultpath = base_runner.run_NSL(args)
    if (args.CIC6s or args.CIC6r or args.CIC16 or args.CIC28):
        full_resultpath = base_runner.run_CIC(args)


    # wrap up
    finishtime = datetime.now()
    print('All done! Wrapping up.\nStart time:\t{}\nFinish time:\t{}\nRuntime:\t{}'.format(starttime, finishtime, finishtime-starttime))
    return full_resultpath


def create_parser():
    """Create an argparse instance that holds all the CLI switches and help"""
    parser = argparse.ArgumentParser(
        description='The MLT either runs data preparation tasks or benchmarks',
        epilog='All arguments are optional - but please remember to choose a dataset when selecting a classifier.',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40)
    )

    # benchmark settings
    bmark = parser.add_argument_group('Benchmark Settings')
    bmark.add_argument('--kfolds', '-k', type=int, default=3, help='Number of folds for validation. Default 3')
    bmark.add_argument('--SingleBenchmark', '--single', action='store_true', help='Used to run a single pass of the benchmark without k-fold CV')
    bmark.add_argument('--unsupervised', '-u', action='store_true', help='Run Benchmark in unsupervised learning mode')

    # dataset utils
    dsutils = parser.add_argument_group('Dataset Preparations and Utilities')
    dsutils.add_argument('--pickleNSL', '--pnsl', action='store_true', help='Create a pickle out of the NSL_KDD dataset')
    dsutils.add_argument('--pickleCICs', '--pcics', action='store_true', help='Create a STRATIFIED pickle out of the CICIDS2017 dataset')
    dsutils.add_argument('--pickleCICr', '--pcicr', action='store_true', help='Create a RANDOMIZED pickle out of the CICIDS2017 dataset')
    dsutils.add_argument('--sanitizeCIC', '--scic', action='store_true', help='Sanitize CICIDS17 and write to _clean.csv')

    # datasets
    datasets = parser.add_argument_group('Dataset Selection')
    datasets.add_argument('--NSL6', '--nsl6', action='store_true', help='Run benchmarks on NSL_KDD with 6 features')
    datasets.add_argument('--NSL16', '--nsl16', action='store_true', help='Run benchmarks on NSL_KDD with 16 features')
    datasets.add_argument('--CIC6s', '--cic6s', action='store_true', help='Run benchmarks on STRATIFIED CICIDS2017 with 6 features')
    datasets.add_argument('--CIC6r', '--cic6r', action='store_true', help='Run benchmarks on RANDOMIZED CICIDS2017 with 6 features')
    datasets.add_argument('--CIC16', '--cic16', action='store_true', help='Run benchmarks on extended CICIDS2017 with 16 features')
    datasets.add_argument('--CIC28', '--cic28', action='store_true', help='Run benchmarks on extended CICIDS2017 with 28 features')

    # implementations
    impls = parser.add_argument_group('Classifier Implementations')
    impls.add_argument('--XGBoost', '--xgb', type=float, nargs=3, metavar=('estimators', 'max_depth', 'lr'), help='Run benchmark on XGBoost. Params: # of estimators and max_depth')
    impls.add_argument('--RandomForest', '--rf', type=float, nargs=2, metavar=('estimators', 'max_depth'), help='Run benchmark on Random Forest. Params: # of estimators and max_depth')
    impls.add_argument('--LSTM2', '--lstm2', type=float, nargs=3, metavar=('batch', 'epochs', 'lr'), help='Run benchmark on a custom 2-class-LSTM')
    impls.add_argument('--HBOS', '--hbos', type=float, nargs=3, metavar=('n_bins', 'alpha', 'tol'), help='Run benchmark on HBOS')
    impls.add_argument('--AutoEncoder', '--ae', type=float, nargs=5, metavar=('batch', 'epochs', 'dropout_rate', 'contamination', 'learning_rate'), help='Run benchmark on AutoEncoder')
    impls.add_argument('--IsolationForest', '--if', nargs=4, metavar=('n_estimators', 'contamination', 'max_features', 'bootstrap'), help='Run benchmark on IsolationForest')

    # tools
    tools = parser.add_argument_group('Tools')
    tools.add_argument('--ResultMail', '--mail', action='store_true', help='Send an email with stats on completion')
    tools.add_argument('--ListResults', '--lsr', nargs=2, metavar=('modelname', 'result_path'), help="List all results in the given folder")
    tools.add_argument('--GenerateCVROCs', '--gcvroc', nargs=5, metavar=('modelname', 'resultpath', 'output_filename', 'model_id_list', 'format_string_list'), help='Compile Model Selection CV results into a single ROC for given path')
    return parser


if __name__ == '__main__':
    main()
