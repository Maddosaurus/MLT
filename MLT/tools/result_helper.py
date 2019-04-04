"""Additional tools to simplify and speed up the result evaluation"""
import os
import json

from tools import toolbelt


def list_scores(modelname, top_resultpath):
    """Lists the metrics.json in every subfolder with the call_params, if existing.

    Args:
        modelname (str): Name of the model to evaluate. Used to derive filenames.
        top_resultpath (str): Path to the parent folder with all subresults
    """
    subresults = toolbelt.list_folders(top_resultpath)
    for sub in subresults:
        single_result_path = os.path.join(top_resultpath, sub)
        list_single_score(modelname, single_result_path)

def gen_ltx(modelname, top_resultpath):
    """Generate a LaTeX table from metrics.json in every subfolder with the call_params, if existing.

    Args:
        modelname (str): Name of the model to evaluate. Used to derive filenames.
        top_resultpath (str): Path to the parent folder with all subresults
    """
    print("\n% algo call params\t    precision & recall & F1   +/-  F1.sd & runtime")


    subresults = toolbelt.list_folders(top_resultpath)

    for sub in subresults:
        single_result_path = os.path.join(top_resultpath, sub)
        try:
            _gen_ltx_line(modelname, single_result_path)
        except FileNotFoundError:
            #print("No fitting metrics file found in folder {}".format(single_result_path))
            pass


def _gen_ltx_line(modelname, resultpath):
    """List score of a single result in given folder as LaTeX compatible table.

    Args:
        modelname (str): Name of the model to evaluate. Used to derive filenames.
        resultpath (str): Path to the folder with a test run result
    """
    try:
        with open(os.path.join(resultpath, 'call_parameters.txt')) as cparms:
            call_vals = None
            for line in cparms:
                try:
                    call_vals = line.split('[')[1].split(']')[0].replace("'", "")
                    call_vals = call_vals.split(', ')
                except IndexError:
                    pass # TODO: Find a more efficient alternative
    except FileNotFoundError:
        pass
    
    metrics = toolbelt.read_from_json(os.path.join(
        resultpath,
        modelname+'_metrics.json'
    ))
    cms = toolbelt.read_from_json(os.path.join(
        resultpath,
        modelname+'_cms.json'
    ))

    call_val_str = " & ".join(call_vals)
    print(call_val_str + " & {:4.2f} & {:4.2f} & {:4.2f} $\pm$ {:4.2f} & {}\\\\ % {}".format(
        metrics['precision']['mean'] * 100,
        metrics['recall']['mean'] * 100,
        metrics['f1_score']['mean'] * 100,
        metrics['f1_score']['sd'] * 100,
        metrics['training_time_mean'],
        os.path.basename(resultpath)
    ))


def list_single_score(modelname, resultpath):
    """List score of a single result in given folder.

    Args:
        modelname (str): Name of the model to evaluate. Used to derive filenames.
        resultpath (str): Path to the folder with a test run result
    """
    try:
        with open(os.path.join(resultpath, 'call_parameters.txt')) as cparms:
            print(cparms.read())
    except FileNotFoundError:
        pass


    metrics = toolbelt.read_from_json(os.path.join(
        resultpath,
        modelname+'_metrics.json'
    ))
    cms = toolbelt.read_from_json(os.path.join(
        resultpath,
        modelname+'_cms.json'
    ))

    try:
        auc = metrics['auc']['mean'] * 100
        auc_sd = metrics['auc']['sd'] * 100
        print(u"\nAUC:\t{:4.2f} \u00B1 {:4.2f}".format(auc, auc_sd))
    except KeyError:
        pass # Might not exist!

    print(u"F1:\t{:4.2f} \u00B1 {:4.2f}".format(
        metrics['f1_score']['mean'] * 100, metrics['f1_score']['sd'] * 100))
    print(u"Acc:\t{:4.2f} \u00B1 {:4.2f}".format(
        metrics['acc']['mean'] * 100, metrics['acc']['sd'] * 100))
    print(u"Prec:\t{:4.2f} \u00B1 {:4.2f}".format(
        metrics['precision']['mean'] * 100, metrics['precision']['sd'] * 100))
    print(u"Recall:\t{:4.2f} \u00B1 {:4.2f}".format(
        metrics['recall']['mean'] * 100, metrics['recall']['sd'] * 100))

    cm = cms['absolute']['fold1']
    tpr = (cm[3] / (cm[3] + cm[2])) * 100 # TP / (TP+FN)
    fpr = (cm[1] / (cm[1] + cm[0])) * 100 # FP / (FP+TN)

    print("TPR (First Fold):\t{:4.2f}\nFPR (First Fold):\t{:4.2f}".format(tpr, fpr))

    try:
        print("Mean traning time: {}\nEnd of stats!\n\n".format(metrics['training_time_mean']))
    except KeyError:
        pass # Might not exist!
