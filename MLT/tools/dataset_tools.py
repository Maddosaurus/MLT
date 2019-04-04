"""Miscellaneous dataset tools and helper functions"""
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

def prin_com_analysis(train_data, test_data, variance=0.95):
    print("Beginning PCA")
    pca = PCA(variance)
    pca.fit(train_data)

    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)

    return train_data, test_data

# Normalization and Scaling
def standard_scale(train_data, test_data):
    """Scale given data with a StandardScaler trained on the train data.

    Args:
        train_data (Pandas.DataFrame or Numpy.ndarray): Training data to scale
        test_data (Pandas.DataFrame or Numpy.ndarray): Test data to scale
    Returns:
        train_data, test_data (Numpy.ndarray): The transformed data sets
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def min_max_scale(train_data, test_data):
    """Scale given data with a MinMaxScaler trained on the train data.

    Args:
        train_data (Pandas.DataFrame or Numpy.ndarray): Training data to scale
        test_data (Pandas.DataFrame or Numpy.ndarray): Test data to scale
    Returns:
        train_data, test_data (Numpy.ndarray): The transformed data sets
    """
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def abs_scaler(train_data, test_data):
    """Scale given data with a MaxAbsScaler trained on the train data.

    Args:
        train_data (Pandas.DataFrame or Numpy.ndarray): Training data to scale
        test_data (Pandas.DataFrame or Numpy.ndarray): Test data to scale
    Returns:
        train_data, test_data (Numpy.ndarray): The transformed data sets
    """
    scaler = preprocessing.MaxAbsScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def powertransform_yeoJohnson(train_data, test_data):
    """Transforms given datasets with a Yeo Johnson Powertransform.

    This transformer will train on the training set and then scale both sets, training and test.
    See I.K. Yeo and R.A. Johnson, “A new family of power transformations to improve normality or symmetry.”
    Biometrika, 87(4), pp.954-959, (2000).

    Args:
        train_data (Pandas.DataFrame or Numpy.ndarray): Training data to transform
        test_data (Pandas.DataFrame or Numpy.ndarray): Test data to transform
    Returns:
        train_data, test_data (Numpy.ndarray): The transformed data sets
    """
    print("Beginning PowerTransform")
    pt = preprocessing.PowerTransformer(method='yeo-johnson', copy=True)
    pt.fit(train_data)

    train_data = pt.transform(train_data)
    test_data = pt.transform(test_data)

    return train_data, test_data

# load pandas df pickles from disk
def load_df(filename, folderpath):
    """Helper function to load Dataframes from a given folder"""
    filepath = os.path.join(folderpath, filename + '.pkl')
    return pd.read_pickle(filepath)
