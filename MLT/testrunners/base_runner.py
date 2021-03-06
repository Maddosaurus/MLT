"""Primary entry point for all testrunners."""
import os

from MLT.datasets import NSL
from MLT.datasets import CIC
from MLT.datasets import Splunk
from MLT.tools import toolbelt
from MLT.testrunners import single_benchmark, kfold_runner


def run_NSL(args):
    """Run the benchmark for a feature subset of NSL_KDD.

    Args:
        args (argparse.Namespace): Argument Namespace containing all parameters for the test run

    Returns:
        result_path (string): Full path where the results can be found
    """
    if args.SingleBenchmark:
        folder_ext = '_single'
    else:
        folder_ext = '_cv'

    if args.NSL6:
        kdd_train_data, kdd_test_data, kdd_train_labels, kdd_test_labels = NSL.get_NSL_6class()
        result_path = toolbelt.prepare_folders('NSL_6class' + folder_ext)
    elif args.NSL16:
        kdd_train_data, kdd_test_data, kdd_train_labels, kdd_test_labels = NSL.get_NSL_16class()
        result_path = toolbelt.prepare_folders('NSL_16class' + folder_ext)

    model_savepath = os.path.join(result_path, 'models')
    # also, save parameters with which the runner was called
    toolbelt.write_call_params(args, result_path)

    # convert to numpy.ndarrays...
    kdd_train_data = kdd_train_data.values
    kdd_test_data = kdd_test_data.values

    # ... and run the benchmark!
    if args.SingleBenchmark:
        return single_benchmark.run_benchmark(
            kdd_train_data, kdd_train_labels,
            kdd_test_data, kdd_test_labels,
            result_path, model_savepath, args
        )
    elif args.kfolds:
        return kfold_runner.run_benchmark(
            kdd_train_data, kdd_train_labels,
            result_path, model_savepath, args
        )


def run_CIC(args):
    """Run the benchmark for a feature subset of CICIDS2017.

    Args:
        args (argparse.Namespace): Argument Namespace containing all parameters for the test run
        stratified (boolean, (optional) default=True): Whether to use the straified or the randomized test data set

    Returns:
        result_path (string): Full path where the results can be found
    """
    if args.CIC20:
        cic_runnername = "CIC_20class"
        cic_train_data, cic_test_data, cic_train_labels, cic_test_labels = CIC.get_CIC_Top20()
    elif args.CIC:
        cic_runnername = "CIC"
        cic_train_data, cic_test_data, cic_train_labels, cic_test_labels = CIC.get_CIC()
    elif args.CICt:
        cic_runnername = "CIC_transformed"
        cic_train_data, cic_test_data, cic_train_labels, cic_test_labels = CIC.get_CIC(transformed=True)


    if args.SingleBenchmark:
        cic_runnername += "_single"
    elif args.kfolds:
        cic_runnername += "_cv"

    result_path = toolbelt.prepare_folders(cic_runnername)
    model_savepath = os.path.join(result_path, 'models')

    # also, save parameters with which the runner was called
    toolbelt.write_call_params(args, result_path)

    # convert to numpy.ndarrays
    cic_train_data = cic_train_data.values
    cic_test_data = cic_test_data.values

    # and run the benchmark!
    if args.SingleBenchmark:
        return single_benchmark.run_benchmark(
            cic_train_data, cic_train_labels,
            cic_test_data, cic_test_labels,
            result_path, model_savepath, args
        )
    elif args.kfolds:
        return kfold_runner.run_benchmark(
            cic_train_data, cic_train_labels,
            result_path, model_savepath, args
        )



def run_Splunk(args):
    """Run the benchmark for the custom Splunk dataset.

    Args:
        args (argparse.Namespace): Argument Namespace containing all parameters for the test run

    Returns:
        result_path (string): Full path where the results can be found
    """
    if args.Splunk:
        runnername = "splunk"
        train_data, test_data, train_labels, test_labels = Splunk.get_splunk_full()
    elif args.SplunkR:
        runnername = "splunk_rand"
        train_data, test_data, train_labels, test_labels = Splunk.get_splunk_full_random()

    if args.SingleBenchmark:
        runnername += "_single"
    elif args.kfolds:
        runnername += "_cv"

    result_path = toolbelt.prepare_folders(runnername)
    model_savepath = os.path.join(result_path, 'models')

    # also, save parameters with which the runner was called
    toolbelt.write_call_params(args, result_path)

    # convert to numpy.ndarrays
    train_data = train_data.values
    test_data = test_data.values

    # and run the benchmark!
    if args.SingleBenchmark:
        return single_benchmark.run_benchmark(
            train_data, train_labels,
            test_data, test_labels,
            result_path, model_savepath, args
        )
    elif args.kfolds:
        return kfold_runner.run_benchmark(
            train_data, train_labels,
            result_path, model_savepath, args
        )