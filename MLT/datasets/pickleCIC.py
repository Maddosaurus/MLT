# coding: utf-8

# # Data Loading and Joining

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'CICIDS2017pub'))


def prepare_dataset():
    """Base function for dataset loading and preparation.

    Returns
    --------
        data : tuple
            A tuple consisting of (cic_data (Pandas.DataFrame), cic_labels (Pandas.DataFrame), group_list (List))
    """

    cic_data = pd.DataFrame()

    datafile_names_sorted = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    ]

    for filename in datafile_names_sorted:
        inputFileName = os.path.join(CIC_FOLDER_PATH, filename)
        print('Appending', inputFileName)

        try:
            new_flows = pd.read_csv(
                inputFileName,
                na_values=["Infinity", "NaN"],
                skipinitialspace=True,
            )
            cic_data = cic_data.append(new_flows, ignore_index=True, sort=False)
        except FileNotFoundError:
            print("WARNING: Could not find CSV! Check your path!")
            exit()

    # ### NaN handling
    print("Handling NaN's")
    # First of all, have a look for undefined / NaN fields.
    # Whilst dropping the whole entry because of these might seem counterintuitive from a science/research perspective,
    # this is a valid approach in the domain of network equipment (drop package if something is wrong).
    cic_data.dropna(inplace=True)

    # **Important**: As we have dropped multiple rows, we need to generate a new index for the DataFrame, as the old one has some gaps now!
    cic_data.index = range(len(cic_data))


    # ## At first glance
    print("Dropping always-0-fields")
    # Having a look at the value distribution reveals that the dataset contains multiple fields that only carry zeroes.
    # We might as well drop these as they only pollute the dataset.
    zero_only = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    cic_data.drop(zero_only, axis=1, inplace=True)



    # ## Fixing the labels
    print("Fixing labels")
    cic_data['Label'] = cic_data['Label'].str.replace('.*XSS', 'WebAttackXSS', regex=True)
    cic_data['Label'] = cic_data['Label'].str.replace('.*Brute Force', 'WebAttackBrute Force', regex=True)
    cic_data['Label'] = cic_data['Label'].str.replace('.*Sql Injection', 'WebAttackSql Injection', regex=True)
    cic_data['Label'] = cic_data['Label'].str.replace(' ', '', regex=True)

    labels = cic_data[['Label']]
    cic_data.drop(['Label'], axis=1, inplace=True)


    # ## Label tokenization
    print("Tokenizing labels")
    # As algorithms cannot work with text-labels directly, lets tokenize them!
    number_of_classes = len(labels['Label'].unique())

    # tokenize the LABELS
    label_tokenizer = Tokenizer(num_words=number_of_classes+1, filters='')
    label_tokenizer.fit_on_texts(labels['Label'].unique())

    # Run the fitted tokenizer on the label column and save the encoded data as dataframe
    enc_labels = pd.DataFrame.from_records(label_tokenizer.texts_to_sequences(labels['Label']), columns=['label_encoded'])

    # To be able to translate the encoded labels back, write the Tokenizer wordlist to a file near the CSVs.
    filename = os.path.join(CIC_FOLDER_PATH, 'cic_label_wordindex.json')
    print('Writing encoder data to file {}:\n\t{}'.format(filename, label_tokenizer.word_index))
    with open(filename, 'w') as outfile:
        json.dump(label_tokenizer.word_index, outfile)


    # Make sure that the shape and index of both data structures adds up, then join them together
    assert enc_labels.shape[0] == labels.shape[0]
    assert list(enc_labels.index.values) == list(labels.index.values)
    labels = pd.concat([labels, enc_labels], axis=1, sort=False)


    # ## Converting Destination Port Information
    print("Converting and encoding Dest Port info")
    # The idea is to group source and destination ports into three categories:
    #   - 0 - 1023:     system / well known ports
    #   - 1024 - 49151: user / registered ports
    #   - \> 49151:     dynamic / private ports
    #
    #
    # Whereas the well known ports will stay untouched, whilst registered and dynamic ports will be converted into their category.
    # set every port to -2 if it is greater than 49151 - reflecting dynamic ports
    cic_data['Destination Port'].where(cic_data['Destination Port'] < 49151, -1, inplace=True)

    # set every port to -1 if it is greater than 1024 - reflecting registered ports
    cic_data['Destination Port'].where(cic_data['Destination Port'] < 1023, -2, inplace=True)



    # ## OHE Destination Ports

    # The ports are a categorical feature that should be used by the ML algos, so it needs to be OHE.
    # To lower the class count, we've already established two meta-cagegories, reserved and dynamic ports.
    def one_hot_encode_drop(dframe, column, prefix):
        dums = pd.get_dummies(dframe[column], prefix=prefix)
        dframe.drop(column, axis=1, inplace=True)
        dframe = pd.concat([dframe, dums], axis=1, sort=False)
        return dframe

    cic_data = one_hot_encode_drop(cic_data, 'Destination Port', 'Destination Port')


    return (cic_data, labels)





def pickleCIC_randomized():
    """Pulls a randomized test partition and pickles it to disk."""
    cic_data, cic_labels = prepare_dataset()

    # ## Train / Test Split
    train_data, test_data, train_labels, test_labels = train_test_split(cic_data, cic_labels, test_size=0.33, random_state=0)


    # Just to be sure that there were no dumb mistakes made, double check the sizes.
    # Shape 0 is the number of entries, which absolutely should match!
    assert test_labels.shape[0] == test_data.shape[0]
    assert train_labels.shape[0] == train_data.shape[0]

    assert list(train_data.index.values) == list(train_labels.index.values)
    assert list(test_data.index.values) == list(test_labels.index.values)


    print("\nNo of train flows:", len(train_data))
    print("No of train labels:", len(train_labels))
    print("Train Label classes: ", train_labels['Label'].unique())
    print("Encoded Train Label classes: ", train_labels['label_encoded'].unique())
    print("-------------------")
    print("No of test flows:", len(test_data))
    print("No of test labels:", len(test_labels))
    print("Test Label classes: ", test_labels['Label'].unique())
    print("Encoded Test Label classes: ", test_labels['label_encoded'].unique())

    # ## Feature Standardization
    # Standartization and Scaling are done while k-fold crossval!

    print('Serializing to disk...')
    # ## Serialization

    # So at this point, we have training and test sets with data and labels. The data parts are encoded and scaled, the encoded indizes are written away as json files.
    # It would be nice if this data could be used for future runs, right? Right!
    # That's why we serialize each dataframe into a python binary pickle on it's own (which is a feature directly supported by [Pandas](https://pandas.pydata.org/pandas-docs/stable/api.html#id12) - nice, eh?)

    # Serialize the result to disk!
    train_data.to_pickle(os.path.join(CIC_FOLDER_PATH, 'cic_train_data_rand.pkl'))
    train_labels.to_pickle(os.path.join(CIC_FOLDER_PATH, 'cic_train_labels_rand.pkl'))

    test_data.to_pickle(os.path.join(CIC_FOLDER_PATH, 'cic_test_data_rand.pkl'))
    test_labels.to_pickle(os.path.join(CIC_FOLDER_PATH, 'cic_test_labels_rand.pkl'))

    print('Finished randomized CICIDS2017 serialization')
