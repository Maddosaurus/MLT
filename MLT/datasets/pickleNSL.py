import os
import json
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
pd.set_option('display.max_columns', None)

# # Pickle NSL-KDD

# This notebooks intended use is to load the CSV data into a Pandas dataframe, normalize and scale the data, then write the DataFrame into a pickle to save these steps for every ML framework run.
# The output are four pickle files: kdd_train_data, kdd_train_labels, kdd_test_data and kdd_test_labels.
# These pickles can be restored as dataframes by calling [pandas.read_pickle()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_pickle.html).

# this yields the FQ path to the folder *datasets*, so it doesn't matter from where the script is called!
NSL_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'NSL_KDD'))

def encode_column_to_int(all_data, dataframe, column, filename):
    # instantiate and fit a tokenizer on exactly the number of values present in the column
    col_encoder = Tokenizer(num_words=len(all_data[column].unique())+1, filters='')
    col_encoder.fit_on_texts(all_data[column].unique())

    fpath = os.path.join(NSL_FOLDER_PATH, filename)
    print('Writing encoder data to file {}: \n{}'.format(fpath, col_encoder.word_index))
    with open(fpath, 'w') as outfile:
        json.dump(col_encoder.word_index, outfile)

    # then transform the column in question
    encoded_col = pd.DataFrame(columns=[column], dtype=np.int8, data=col_encoder.texts_to_sequences(dataframe[column]))

    #drop the source column
    dataframe.drop([column], axis=1, inplace=True)

    # and append the encoded one to the original dataframe
    dataframe = pd.concat([dataframe, encoded_col], axis=1)
    return dataframe


def write_to_pickle(dataframe, filename):
    dataframe.to_pickle(os.path.join(NSL_FOLDER_PATH, filename+'.pkl'))

# ## Data Loading and Prep

# Fist of all, load the header, so Pandas is able to identify the columns correctly.
# As the header info for the last two columns is missing, I am adding these by hand to the DataFrame.
# They are the type of traffic, *label*, and the difficulty of detection, *difficulty_level*.
# These columns are removed from the data-pickled but for completeness' sake, I'm including these.

def pickle_NSL_KDD():
    print('Loading header info from "Field Names.csv"')
    header_col = pd.read_csv(os.path.join(NSL_FOLDER_PATH, 'Field Names.csv'), header=None)
    header_col = header_col.append(pd.DataFrame([['label', 'symbolic'], ['difficulty_level', 'continuous']]))

    header_names = header_col[0].values
    print('Loaded {} header names:  \n{}'.format(len(header_names), header_names))


    # ### Training Set
    # We're using "KDDTrain+" specifically, as this is the full training data set.

    print('Loading KDDTrain+...')
    ftrain = os.path.join(NSL_FOLDER_PATH, 'KDDTrain+.csv')
    kdd_train_data = pd.read_csv(ftrain, header=None, names=header_names)

    # split off labels
    kdd_train_labels = kdd_train_data.filter(['label', 'difficulty_level'])

    # ...and drop them as well as the difficulty level
    kdd_train_data.drop(['label', 'difficulty_level'], axis=1, inplace=True)

    # ### Test Set
    # As with the Training set, we're using the full "KDDTest+" file.

    print('Loading KDDTest+...')
    ftest = os.path.join(NSL_FOLDER_PATH, 'KDDTest+.csv')
    kdd_test_data = pd.read_csv(ftest, header=None, names=header_names)

    # split off labels..
    kdd_test_labels = kdd_test_data.filter(['label', 'difficulty_level'])

    # ...and drop them as well as the difficulty level
    kdd_test_data.drop(['label', 'difficulty_level'], axis=1, inplace=True)

    print('Encoding String Data...')
    # ## Attack Label Encoding
    # As the labels are still plaintext, they need to be converted to simple integer representations.
    all_labels = pd.concat([kdd_train_labels, kdd_test_labels])

    label_tokenizer = Tokenizer(num_words=len(all_labels['label'].unique())+1, filters='')
    label_tokenizer.fit_on_texts(all_labels['label'].unique())

    filename = os.path.join(NSL_FOLDER_PATH, 'kdd_label_wordindex.json')
    print('Writing encoder data to file {}: {}'.format(filename, label_tokenizer.word_index))
    with open(filename, 'w') as outfile:
        json.dump(label_tokenizer.word_index, outfile)

    # ### Train Data
    encoded_train_labels = pd.DataFrame(columns=['label_encoded'], dtype=np.int8, data=label_tokenizer.texts_to_sequences(kdd_train_labels['label']))
    kdd_train_labels = pd.concat([kdd_train_labels, encoded_train_labels], axis=1)


    # ### Test Data
    encoded_test_labels = pd.DataFrame(columns=['label_encoded'], dtype=np.int8, data=label_tokenizer.texts_to_sequences(kdd_test_labels['label']))
    kdd_test_labels = pd.concat([kdd_test_labels, encoded_test_labels], axis=1)


    # ## Remaining Column Encoding

    # Besides the label, there are three more columns that need to be translated from text to integer data: __protocol_type__, __service__ and __flag__.
    # As these are done the same way, I define a function that fits a tokenizer, transforms the texts and appends the encoded column.
    # The plaintext columns are dropped, as this is needed for normalization. The regarding class indexes are written as JSON to *dataset_columnName_wordindex.json* along with the pickles.


    # create one big dataframe for training the encoders
    all_data = pd.concat([kdd_train_data, kdd_test_data])


    kdd_train_data = encode_column_to_int(all_data, kdd_train_data, 'protocol_type', 'kdd_train_data_protocol_type_wordindex.json')
    kdd_test_data = encode_column_to_int(all_data, kdd_test_data, 'protocol_type', 'kdd_test_data_protocol_type_wordindex.json')

    kdd_train_data = encode_column_to_int(all_data, kdd_train_data, 'service', 'kdd_train_data_service_wordindex.json')
    kdd_test_data = encode_column_to_int(all_data, kdd_test_data, 'service', 'kdd_test_data_service_wordindex.json')

    kdd_train_data = encode_column_to_int(all_data, kdd_train_data, 'flag', 'kdd_train_data_flag_wordindex.json')
    kdd_test_data = encode_column_to_int(all_data, kdd_test_data, 'flag', 'kdd_test_data_flag_wordindex.json')


    # ## Feature Standardization
    # Standardization and Scaling are done inside the k-fold crossval!

    print('Serializing DataFrames to Pickles...')
    # ## Serialization

    # So at this point, we have training and test sets with data and labels. The data parts are encoded  and the encoded indices are written away as json files.
    # It would be nice if this data could be used for future runs, right? Right!
    # That's why we serialize each dataframe into a python binary pickle on it's own (which is a feature directly supported by [Pandas](https://pandas.pydata.org/pandas-docs/stable/api.html#id12) - nice, eh?)

    write_to_pickle(kdd_train_data, 'kdd_train_data')
    write_to_pickle(kdd_test_data, 'kdd_test_data')
    write_to_pickle(kdd_train_labels, 'kdd_train_labels')
    write_to_pickle(kdd_test_labels, 'kdd_test_labels')


    # These can be loaded into a Pandas DataFrame like this:
    # `someDataFrame = pd.read_pickle("./dummy.pkl")`

    print('Finished pickling of NSL_KDD')


if __name__ == '__main__':
    pickle_NSL_KDD()
