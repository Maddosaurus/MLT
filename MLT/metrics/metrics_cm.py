"""Utility module for generating various confusion matrix flavours."""
import os
import json
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def save_cm_arr_to_disk(cm_array, modelname, result_path):
    """Save the confusion array for a given model to disk as a json"""
    cms = {}
    cms["legend"] = ['tn', 'fp', 'fn', 'tp']
    cms["absolute"] = {}
    cms["relative"] = {}

    for idx, cmatrix in enumerate(cm_array):
        fold_id = 'fold{}'.format(idx+1)
        cms['absolute'][fold_id] = cmatrix.ravel().tolist()
        cms['relative'][fold_id] = normalize_cm(cmatrix).ravel().tolist()

    print("\nConfusion Matrices:\n")
    print(json.dumps(cms, indent=4))

    filepath = os.path.join(result_path, modelname + '_cms.json')
    with open(filepath, 'w') as mjson:
        json.dump(cms, mjson)


def normalize_cm(cmatrix):
    """Translate absolte values of a CM to relative distributions"""
    return cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]

def calc_cm(prediction_data):
    """Calculate the Confusion Matrices for given prediction entries"""
    all_cm = []
    for pred in prediction_data:
        all_cm.append(confusion_matrix(pred.test_labels, pred.predicted_labels))
    return all_cm


def generate_all_cm_to_disk(cm_array, modelname, filepath):
    """Generate normalized and absolute matrices as images at the given filepath"""
    for index, cmmatrix in enumerate(cm_array):
        loop_filename = modelname + "_Fold_" + str(index+1)
        loop_filename_norm = loop_filename+"_normalized"
        generate_confusion_matrix_to_disk(
            cmmatrix,
            ['Benign', 'Attack'],
            loop_filename,
            filepath
        )
        generate_confusion_matrix_to_disk(
            cmmatrix,
            ['Benign', 'Attack'],
            loop_filename_norm,
            filepath,
            True
        )


# slightly altered version of
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def generate_confusion_matrix_to_disk(cmatrix, classes, modelname, filepath, normalize=False):
    """Plot and save the confusion matrix as a picture.

    Classes are fixed and given, as well as the save path and the modelname.
    The latter also gets incorporated in the plot title.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix for ' + modelname)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(
            j, i,
            format(cmatrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if cmatrix[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    savepath = os.path.join(filepath, modelname + '.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close('all')
