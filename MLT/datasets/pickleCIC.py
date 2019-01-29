import os
import sys
import traceback
import json
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tools import toolbelt


pd.set_option('display.max_columns', None)

# # Pickle CICIDS 2017

# This notebooks intended use is to load the CSV data into a Pandas dataframe, normalize and scale the data, then write the DataFrame into a pickle to save these steps for every ML framework run.
# The output are four pickle files: cic_train_data, cic_test_data, cic_train_labels and cic_test_labels.
# **TODO**: Split these into training and test in a meaningful way! (Most likely by hand?)
# These pickles can be restored as dataframes by calling [pandas.read_pickle()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_pickle.html).
# **Hint**: If your are missing the *_clean* CSVs, try running the notebook *Data Sanitazation.ipynb*

CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'CICIDS2017'))


def write_to_pickle(dataframe, filename):
    dataframe.to_pickle(os.path.join(CIC_FOLDER_PATH, filename+'.pkl'))


def is_same_comm(curr, prev):
    if prev['source_ip_o3'] == curr['source_ip_o3'] and \
        prev['source_ip_o4'] == curr['source_ip_o4'] and \
        prev['destination_ip_o3'] == curr['destination_ip_o3'] and \
        prev['destination_ip_o4'] == curr['destination_ip_o4'] and \
        prev['source_port'] == curr['source_port'] and \
        prev['destination_port'] == curr['destination_port']:
        return True
    else:
        return False


def is_bidirectional_comm(curr, prev):
    if prev['source_ip_o3'] == curr['destination_ip_o3'] and \
        prev['source_ip_o4'] == curr['destination_ip_o4'] and \
        prev['destination_ip_o3'] == curr['source_ip_o3'] and \
        prev['destination_ip_o4'] == curr['source_ip_o4'] and \
        prev['source_port'] == curr['destination_port'] and \
        prev['destination_port'] == curr['source_port']:
        return True
    else:
        return False


def prepare_dataset():
    """Base function for dataset loading and preparation.
    
    Returns
    --------
        data : tuple
            A tuple consisting of (cic_data (Pandas.DataFrame), cic_labels (Pandas.DataFrame), group_list (List))
    """
    # ## Data Loading and Prep

    # As there is literally "Inifnity" written in the CSV dataset, we set an additional filter so that these will be replaced by a NaN-representation (that will lateron be set to zero).
    # Also, the external_ip field is set to 0.0.0.0 if either NaN or non existent.
    # Finally, the dtype for the external_ip column had to be set manually to object as Pandas kept getting confused.


    cic_data = pd.DataFrame()

    # if you are missing these CSVs, run `main.py --sanitizeCIC` to generate them
    datafile_names_sorted = [
        'Monday-WorkingHours.pcap_ISCX_clean.csv',
        'Tuesday-WorkingHours.pcap_ISCX_clean.csv',
        'Wednesday-WorkingHours.pcap_ISCX_clean.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_clean.csv',
        'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_clean.csv',
        'Friday-WorkingHours-Morning.pcap_ISCX_clean.csv',
        'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_clean.csv',
        'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_clean.csv'
    ]

    for filename in datafile_names_sorted:
        inputFileName = os.path.join(CIC_FOLDER_PATH, filename)
        print('Appending', inputFileName)

        try:
            new_flows = pd.read_csv(
                inputFileName,
                na_values="Infinity",
                dtype={'external_ip':'object'},
                parse_dates=['timestamp']
            )
        except FileNotFoundError:
            print("WARNING: Could not find clean versions of CSV! Check Path or try running the cleanup scripts first!")
            exit()

        # as this field is not in all flows, double check for it
        if 'external_ip' not in new_flows:
            new_flows['external_ip'] = "0.0.0.0"
        new_flows['external_ip'].fillna("0.0.0.0", inplace=True)

        cic_data = cic_data.append(new_flows, ignore_index=True, sort=False)

    print('Found these class labels:', str(cic_data.label.unique()))


    # Replace NA-values with 0
    cic_data.fillna(value=0, inplace=True)
    print('Null-vars present in dataset?', cic_data.isnull().values.any())

    print('Encoding String data...')
    # ## Data Encoding

    # There's still a problem: How can we encode IP addresses in a way that the neural network can make use of them while preserving the hierarchical information they contain?
    # Encoding IPs through One Hot lets comlexity and training times explode, so for now I am splitting each IP into its four octet pairs and interpret them as numbers.
    # Maybe there's a better way to represent them (especially because I am only able to encode IPv4 right now).
    #
    # **Important**: If this breaks, you forgot to remove the broken external IP in Friday DDoS @ 2017-07-07T15:58:00,26794

    # https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns
    # FIXME: Right now, only IPv4 (4 octets)

    try:
        # Split the String representation of the IP into it's four octects, which are delimited by a dot
        cic_data['source_ip_o1'], cic_data['source_ip_o2'], cic_data['source_ip_o3'], cic_data['source_ip_o4'] = cic_data['source_ip'].str.split('.').str
        cic_data['destination_ip_o1'], cic_data['destination_ip_o2'], cic_data['destination_ip_o3'], cic_data['destination_ip_o4'] = cic_data['destination_ip'].str.split('.').str
        cic_data['external_ip_o1'], cic_data['external_ip_o2'], cic_data['external_ip_o3'], cic_data['external_ip_o4'] = cic_data['external_ip'].str.split('.').str
    except ValueError:
        print('Encountered Value error! Have you removed the broken external IP entry in Friday DDoS @ 2017-07-07T15:58:00,26794?')
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    # After completion, drop the initial columns, as they aren't needed anymore
    cic_data.drop(['source_ip'], axis=1, inplace=True)
    cic_data.drop(['destination_ip'], axis=1, inplace=True)
    cic_data.drop(['external_ip'], axis=1, inplace=True)

    # Also, set the new fields to int, as they are to be interpreted as such
    cic_data['source_ip_o1'] = cic_data.source_ip_o1.astype(int)
    cic_data['source_ip_o2'] = cic_data.source_ip_o2.astype(int)
    cic_data['source_ip_o3'] = cic_data.source_ip_o3.astype(int)
    cic_data['source_ip_o4'] = cic_data.source_ip_o4.astype(int)
    cic_data['destination_ip_o1'] = cic_data.destination_ip_o1.astype(int)
    cic_data['destination_ip_o2'] = cic_data.destination_ip_o2.astype(int)
    cic_data['destination_ip_o3'] = cic_data.destination_ip_o3.astype(int)
    cic_data['destination_ip_o4'] = cic_data.destination_ip_o4.astype(int)
    cic_data['external_ip_o1'] = cic_data.external_ip_o1.astype(int)
    cic_data['external_ip_o2'] = cic_data.external_ip_o2.astype(int)
    cic_data['external_ip_o3'] = cic_data.external_ip_o3.astype(int)
    cic_data['external_ip_o4'] = cic_data.external_ip_o4.astype(int)


    # Now that this is out of the way, we still need to encode the labels column to numeric values.
    # To do this, I'm going to be using the Keras Tokenizer.
    # The labels of the dataset (as in: *Benign*, *DDoS*, *Portscan*, etc) are converted into a list of integers and split off of the main DataFrame.
    #
    # So if the order of the first three Netflows would be *Benign*, *Benign*, *DDos*,
    # the result would look like this: `[1,1,2]`

    cic_labels = cic_data.filter(['label'])
    cic_data.drop(['label'], axis=1, inplace=True)


    number_of_classes = len(cic_labels['label'].unique())

    # tokenize the LABELS
    label_tokenizer = Tokenizer(num_words=number_of_classes+1, filters='')
    label_tokenizer.fit_on_texts(cic_labels['label'].unique())

    # Run the fitted tokenizer on the label column and save the encoded data as dataframe
    enc_labels = label_tokenizer.texts_to_sequences(cic_labels['label'])

    # finally, append the encoded labels to the label dataframe
    cic_labels = pd.concat([cic_labels, pd.DataFrame(columns=['label_encoded'], dtype=np.int8, data=enc_labels)], axis=1)


    # To be able to translate the encoded labels back, write the Tokenizer wordlist to a file near the CSVs.
    filename = os.path.join(CIC_FOLDER_PATH, 'cic_label_wordindex.json')
    print('Writing encoder data to file {}: {}'.format(filename, label_tokenizer.word_index))
    with open(filename, 'w') as outfile:
        json.dump(label_tokenizer.word_index, outfile)

    # Create grouping, if it is not serialized. If it is, skip!
    grouping_file = os.path.join(CIC_FOLDER_PATH, 'cic_grouping.pkl')
    if os.path.exists(grouping_file):
        print("Found a serialized grouping file. Skipping grouping!")
        group_list = toolbelt.read_from_pickle(grouping_file)

    else:
        print("Beginning grouping of the dataset!")
        same_bi_comm_counter = 0

        group_list = []
        group_list_index = 0
        group_list.append([])
        prev = None

        for index, row in cic_data.iterrows():
            if index == 0:
                prev = cic_data.iloc[0]
            else:
                prev = cic_data.iloc[index-1]

            curr = cic_data.iloc[index]

            # TODO: Can we make use of the pandas grouping function here?
            if is_same_comm(curr, prev) or is_bidirectional_comm(curr, prev):
                same_bi_comm_counter += 1
            else:
                group_list_index += 1
                group_list.append([])

            group_list[group_list_index].append({'cic_index': index, 'label': cic_labels.iloc[index]['label'], 'label_encoded': cic_labels.iloc[index]['label_encoded']})

        # Write groupings to file
        toolbelt.write_to_pickle(grouping_file, group_list)
        print('I parsed {} entries, whereof {} were sequential or bidirectional'.format(len(cic_data.index), same_bi_comm_counter))

    print("\nGrouped list: ")
    print("List of Lists legth: {}".format(len(group_list)))

    return (cic_data, cic_labels, group_list)



def pickleCIC_stratified():
    """Pulls a stratified test partition and pickles it to disk."""
    cic_data, cic_labels, group_list = prepare_dataset()


    labels_enc = []

    for idx, glist in enumerate(group_list):
        labels_enc.append(glist[0]['label_encoded'])

    np_group_list = np.array(group_list)
    np_labels = np.array(labels_enc)

    skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=0)

    # not nice, but cannot find another way quickly to only get 1 fold
    index = 0
    train = []
    test = []
    for train_index, test_index in skf.split(np_group_list, np_labels):
        if index > 0:
            continue
        print("Train:", train_index, "Test:", test_index)
        train = train_index
        test = test_index
        index += 1

    # What we need to do next:
    # Take these lists from the fold, join their indices and then copy from the single dataframe (and the label df) 
    # into train and testpartition
    train_indices = []
    for entry in train:
        centry = np_group_list[entry]
        for member in centry:
            train_indices.append(member['cic_index'])
            
    test_indices = []
    for entry in test:
        centry = np_group_list[entry]
        for member in centry:
            test_indices.append(member['cic_index'])

    # write the chosen indices to disk
    toolbelt.write_to_json(os.path.join(CIC_FOLDER_PATH, "train_test_indices.json"), {"train_indices": train_indices, "test_indices": test_indices})

    train_data_df = cic_data.iloc[train_indices]
    train_labels_df = cic_labels.iloc[train_indices]
    test_data_df = cic_data.iloc[test_indices]
    test_labels_df = cic_labels.iloc[test_indices]

    print("\nNo of train flows:", len(train_data_df))
    print("No of train labels:", len(train_labels_df))
    print("Train labels:", train_labels_df.label_encoded.unique())
    print("-------------------")
    print("No of test flows:", len(test_data_df))
    print("No of test labels:", len(test_labels_df))
    print("Test labels:", test_labels_df.label_encoded.unique())

    # ## Feature Standardization
    # Standartization and Scaling are done while k-fold crossval!

    print('Serializing to disk...')
    # ## Serialization

    # So at this point, we have training and test sets with data and labels. The data parts are encoded and scaled, the encoded indizes are written away as json files.
    # It would be nice if this data could be used for future runs, right? Right!
    # That's why we serialize each dataframe into a python binary pickle on it's own (which is a feature directly supported by [Pandas](https://pandas.pydata.org/pandas-docs/stable/api.html#id12) - nice, eh?)

    write_to_pickle(train_data_df, 'cic_train_data_stratified')
    write_to_pickle(train_labels_df, 'cic_train_labels_stratified')
    write_to_pickle(test_data_df, 'cic_test_data_stratified')
    write_to_pickle(test_labels_df, 'cic_test_labels_stratified')

    print('Finished stratified CICIDS2017 serialization')




def pickleCIC_randomized():
    """Pulls a randomized test partition and pickles it to disk."""
    cic_data, cic_labels, group_list = prepare_dataset()


    labels_enc = []

    for idx, glist in enumerate(group_list):
        labels_enc.append(glist[0]['label_encoded'])

    np_group_list = np.array(group_list)
    np_labels = np.array(labels_enc)
    np_indices = np.arange(len(np_group_list)) # indices! see https://stackoverflow.com/questions/31521170/scikit-learn-train-test-split-with-indices/31535238#31535238

    # the labels are technically not needed and could be ommited
    train_data, test_data, train_labels, test_labels, index_train, index_test = train_test_split(np_group_list, np_labels, np_indices, test_size=0.33, random_state=0)


    # What we need to do next:
    # Take these lists from the fold, join their indices and then copy from the single dataframe (and the label df) 
    # into train and testpartition
    train_indices = []
    for entry in index_train:
        centry = np_group_list[entry]
        for member in centry:
            train_indices.append(member['cic_index'])
            
    test_indices = []
    for entry in index_test:
        centry = np_group_list[entry]
        for member in centry:
            test_indices.append(member['cic_index'])

    # write the chosen indices to disk
    toolbelt.write_to_json(os.path.join(CIC_FOLDER_PATH, "train_test_indices.json"), {"train_indices": train_indices, "test_indices": test_indices})

    train_data_df = cic_data.iloc[train_indices]
    train_labels_df = cic_labels.iloc[train_indices]
    test_data_df = cic_data.iloc[test_indices]
    test_labels_df = cic_labels.iloc[test_indices]

    print("\nNo of train flows:", len(train_data_df))
    print("No of train labels:", len(train_labels_df))
    print("Train labels:", train_labels_df.label_encoded.unique())
    print("-------------------")
    print("No of test flows:", len(test_data_df))
    print("No of test labels:", len(test_labels_df))
    print("Test labels:", test_labels_df.label_encoded.unique())

    # ## Feature Standardization
    # Standartization and Scaling are done while k-fold crossval!

    print('Serializing to disk...')
    # ## Serialization

    # So at this point, we have training and test sets with data and labels. The data parts are encoded and scaled, the encoded indizes are written away as json files.
    # It would be nice if this data could be used for future runs, right? Right!
    # That's why we serialize each dataframe into a python binary pickle on it's own (which is a feature directly supported by [Pandas](https://pandas.pydata.org/pandas-docs/stable/api.html#id12) - nice, eh?)

    write_to_pickle(train_data_df, 'cic_train_data_randomized')
    write_to_pickle(train_labels_df, 'cic_train_labels_randomized')
    write_to_pickle(test_data_df, 'cic_test_data_randomized')
    write_to_pickle(test_labels_df, 'cic_test_labels_randomized')

    print('Finished randomized CICIDS2017 serialization')
