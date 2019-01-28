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
        print("##############################")
        single_result_path = os.path.join(top_resultpath, sub)
        list_single_score(modelname, single_result_path)



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
    print("Scores:", json.dumps(metrics, indent=4))

    cms = toolbelt.read_from_json(os.path.join(
        resultpath,
        modelname+'_cms.json'
    ))

    auc = metrics['auc']['mean'] * 100
    auc_sd = metrics['auc']['sd'] * 100
    f1 = metrics['f1_score']['mean'] * 100
    f1_sd = metrics['f1_score']['sd'] * 100
    acc = metrics['acc']['mean'] * 100
    acc_sd = metrics['acc']['sd'] * 100
    
    print("AUC:\t{:4.2f} +/- {:4.2f}".format(auc, auc_sd))
    print("F1:\t{:4.2f} +/- {:4.2f}".format(f1, f1_sd))
    print("Acc:\t{:4.2f} +/- {:4.2f}".format(acc, acc_sd))
    
    cm = cms['absolute']['fold1']
    tpr = (cm[3] / (cm[3] + cm[2])) * 100 # TP / (TP+FN)
    fpr = (cm[1] / (cm[1] + cm[0])) * 100 # FP / (FP+TN)

    print("TPR (First Fold):\t{:4.2f}\nFPR (First Fold):\t{:4.2f}".format(tpr, fpr))

    print("Mean traning time: {}".format(metrics['training_time_mean']))
