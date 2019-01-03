"""Utility functions for generating ROC and AUC statistics"""
import os
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

from tools import toolbelt


def calc_auc(prediction_data):
    """Calculated the area under curve on given DF"""
    all_auc = []
    for pred in prediction_data:
        all_auc.append(roc_auc_score(pred.test_labels, pred.predicted_probabilities))
    return all_auc


# This is a slightly adapted version of
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def generate_avg_roc_to_disk(prediction_data, modelname, filepath):
    """Generates an average of all given ROCs and plots all ROCs and Avg to a single figure"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 1

    for pred in prediction_data:
        fpr, tpr, tresholds = roc_curve(pred.test_labels, pred.predicted_probabilities)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC: %0.2f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC: %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for {}'.format(modelname))
    plt.legend(loc="lower right")

    savepath = os.path.join(filepath, modelname + '.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close('all')


def append_roc_model_selection(result_json, modelname, line_format):
    """Appends the CV-mean ROC to an existing plot."""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for pred in result_json:
        fpr, tpr, tresholds = roc_curve(pred["test_labels"], pred["predicted_probabilities"])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, line_format,
             markevery=5,                       # only plot every 5th marker to not overcrowd the graph
             label='%s\n (AUC: %0.2f $\pm$ %0.2f)' % (modelname, mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)


def generate_cv_roc_model_selection(modelname, result_path, parameter_name, model_id_list=None, format_list=None):
    """Generates an average of all CV-results in a given folder.

    Point this function to a folder that contains multiple result-subfolders with crossvalidated results.
    It will generate the average ROC for every result and add them all to a single figure.

    Args:
        modelname (string): The name of the model to draw. Will be used to determine filename and title of the plot.
        result_path (string or list): Path to the result base folder that contains multiple test runs. Can be a list of single runs. All runs will be combined into a single figure.
        parameter_name (string): Parameter under test - will be in the title and appended to the filename.
        model_id_list (list): A list of Strings. This is used for the legend.
        format_list (list): A list of pyplot format Strings to be used for the single plots.
    """
    if len(result_path) > 1:
        subfolders = result_path
        result_path = os.path.join(subfolders[0], '..')
    else:
        result_path = result_path[0]
        subfolders = toolbelt.list_folders(result_path)
    
    no_of_subs = len(subfolders)

    # define a list of line style format strings for the different plots.
    # See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    if format_list is None or len(format_list) < no_of_subs:
        print('Format not provided or not enough entries. Falling back to default\nRemember to clean old results!')
        format_list = ['ks--', 'kv--', 'k<--', 'kp--', 'kx--', 'kd--', 'bs--', 'bv--', 'b<--', 'bp--', 'bx--', 'bd--']
    if model_id_list is None or len(model_id_list) < no_of_subs:
        print('Model names not provided or not enough entries. Falling back to default\nRemember to clean old results!')
        model_id_list = []
        for _ in range(20):
            model_id_list.append(modelname) # Just use the provided modelname

    # Clear all remaining plots
    plt.clf()
    plt.close('all')

    # add luck
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    
    # loop over subfolders and generate the ROCs
    for idx, subfolder in enumerate(subfolders):
        cv_runpath = os.path.join(result_path, subfolder)
        cv_results = toolbelt.load_results_from_disk(cv_runpath, modelname)
        print("Loaded runpath: {}".format(cv_runpath))
        append_roc_model_selection(cv_results, model_id_list[idx], format_list[idx])

    # set main legend and info
    # Legend placement: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ({})'.format(modelname, parameter_name))
    plt.legend(loc="lower right", ncol=2, prop={'size': 8})

    # save to disk
    folder_name = (os.path.basename(os.path.realpath(result_path)))
    savepath = os.path.join(result_path, folder_name + '-' + modelname + '_' + parameter_name + '.pdf')
    plt.savefig(savepath, bbox_inches="tight", format="pdf")
    
    #Finally, clean again
    plt.clf()
    plt.close('all')