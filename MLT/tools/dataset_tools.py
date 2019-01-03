"""Miscellaneous dataset tools and helper functions"""
import os
import pandas as pd
from sklearn import preprocessing

# Normalization and Scaling
def normalize_and_scale(train_data, test_data):
    """Normalize and scale given data with a scaler trained on the train data."""
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data

# load NSL pandas df pickles from disk
def load_df(filename, folderpath):
    """Helper function to load Dataframes from a given folder"""
    filepath = os.path.join(folderpath, filename + '.pkl')
    return pd.read_pickle(filepath)
