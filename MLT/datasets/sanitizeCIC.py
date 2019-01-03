import csv
import os
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)

# ## The Problem

# Unfortunately, the CSV data of CICIDS 2017 has some problems that need to be tackled before being able to reliably use it.
# The main problems:
# - Monday has a different timestamp format as the other days
# - The timestamps are neither in 24h format nor in 12h AM/PM format, rendering all dataset parts after noon unusable
# - The CSV headers contain spaces (middle/leading/training) and special chars like forward slashes
# - Thursday morning contains empty datasets for the last 288.602 entries
#

CIC_FOLDER_PATH = (os.path.join(os.path.dirname(__file__), 'CICIDS2017'))

def calc_timestamp(timestamp):
    new_time = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S')

    # convert any hour between 13 and 17 o'clock into their 24h-pendant
    # as the captures run from 08:00 to 17:00 o'clock
    if new_time.hour in range(1, 8):
        hour24 = new_time.hour + 12
        new_time = new_time.replace(hour=hour24)
    return new_time.isoformat()


def calc_new_timestamp(timestamp):
    new_time = datetime.strptime(timestamp, '%d/%m/%Y %H:%M')

    # convert any hour between 13 and 17 o'clock into their 24h-pendant
    # as the captures run from 08:00 to 17:00 o'clock
    if new_time.hour in range(1, 8):
        hour24 = new_time.hour + 12
        new_time = new_time.replace(hour=hour24)
    return new_time.isoformat()


def calc_fri_timestamp(timestamp):
    new_time = datetime.strptime(timestamp, '%d/%m/%Y %H:%M')

    # convert any hour between 13 and 17 o'clock into their 24h-pendant
    # as the captures run from 08:00 to 17:00 o'clock
    if new_time.hour in range(1, 8):
        hour24 = new_time.hour + 12
        new_time = new_time.replace(hour=hour24)
    return new_time.isoformat()


# The timestamp format that is used on monay isn't used on the other days.
# Instad, an inconsistent format string is applied that leaves afternoon data unusable.
# The need for data sanitization is given.

# As we need to treat every file a bit different, 
# there is no easy way of automating the needed transformations.
# As I want to stay as interoperable as possible, 
# I am going to convert all timestamps to ISO8601 format.
# Before transforming the data itself, it would be advisable to build a working CSV header

# ### Towards a sane CSV header

# We are going to transform the header into a more useable format. 
# The following steps will be applied:
# - Remove leading and trailing whitespace
# - Convert all colum names to lowercase
# - Replace connecting whitespaces with underscores
# - Replace forward slashed with `_per_` to avoid dealing with encoding or escaping issues

def sanitize_CICIDS2017():

    header = "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label"
    header = header.split(',')
    header = list(map(str.strip, header))
    header = list(map(str.lower, header))
    header = list(map(lambda s: s.replace(' ', '_'), header))
    header = list(map(lambda s: s.replace('/', '_per_'), header))
    ddos_header = header
    ddos_header.append('external_ip')
    print(",".join(ddos_header))


    # ### Updating the headers

    # We are now ready to update the headers in the original CSV files.
    # As we don't want to touch originals, 
    # we are creating a new file with the extension `_clean` on which we will continue to work

    for filename in os.listdir(CIC_FOLDER_PATH): # this is where the CIC CSV files live
        if filename.endswith('.csv'):
            if filename.endswith('_clean.csv'):
                continue    # skip already cleaned files
            inputFileName = os.path.join(CIC_FOLDER_PATH, filename)
            outputFileName = os.path.join(CIC_FOLDER_PATH, os.path.splitext(filename)[0] +"_clean.csv")
            print("Opening", inputFileName)

            with open(inputFileName, newline='', encoding="utf-8", errors="ignore") as inputFile, open(outputFileName, 'w', newline='', encoding="utf-8", errors="ignore") as outfile:

                r = csv.reader(inputFile)
                w = csv.writer(outfile)

                print("Writing new header")
                next(r, None) # skip the header of the original file
                w.writerow(header)

                print("Copying remaining file")
                for row in r:
                    w.writerow(row)


    # ## Transforming the Datasets
    print("Beginning conversion of Monday data")
    # ### Monday
    # The monday timestamps are best converted to a datetimesamp and then updated accordingly.

    # open the cleaned version with sane headers. This time, open the full csv
    monday_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Monday-WorkingHours.pcap_ISCX_clean.csv'), dtype=str)

    # Now convert every single timestamp
    monday_df['timestamp'] = monday_df.timestamp.apply(calc_timestamp)

    # If everything went well, save changes to disk
    monday_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Monday-WorkingHours.pcap_ISCX_clean.csv'), index=False)


    print("Beginning conversion of Tuesday data")
    # ### Tuesday
    # The tuesday timestamps are different from mondays', so we need to update the date function.
    tuesday_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Tuesday-WorkingHours.pcap_ISCX_clean.csv'))

    # The timestamp needs fixing.
    tuesday_df['timestamp'] = tuesday_df.timestamp.apply(calc_new_timestamp)

    # Besides the timestamp, the attack labels don't look too good either

    tuesday_df['label'] = tuesday_df['label'].str.replace('FTP-Patator', 'FTPPatator')
    tuesday_df['label'] = tuesday_df['label'].str.replace('SSH-Patator', 'SSHPatator')

    # If everything went well, save changes to disk
    tuesday_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Tuesday-WorkingHours.pcap_ISCX_clean.csv'), index=False)


    print("Beginning conversion of Wednesday data")
    # ### Wednesday

    # open the cleaned version with sane headers. This time, open the full csv
    wednesday_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Wednesday-WorkingHours.pcap_ISCX_clean.csv'), dtype=str)

    wednesday_df['timestamp'] = wednesday_df.timestamp.apply(calc_new_timestamp)


    # Besides the timestamp, the attack labels don't look too good either
    wednesday_df['label'].unique()

    wednesday_df['label'] = wednesday_df['label'].str.replace('DoS slowloris', 'DoSSlowloris')
    wednesday_df['label'] = wednesday_df['label'].str.replace('DoS Slowhttptest', 'DoSSlowhttptest')
    wednesday_df['label'] = wednesday_df['label'].str.replace('DoS Hulk', 'DoSHulk')
    wednesday_df['label'] = wednesday_df['label'].str.replace('DoS GoldenEye', 'DoSGoldenEye')


    # If everything went well, save changes to disk
    wednesday_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Wednesday-WorkingHours.pcap_ISCX_clean.csv'), index=False)


    print("Beginning conversion of Thursday data")
    # ### Thursday

    # Thursday is split into two files - with the morning dataset having issues that values are empty.
    # These empty lines have been trimmed manually.

    # #### Morning
    # open the cleaned version with sane headers. This time, open the full csv
    thursday_morn_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_clean.csv'), dtype=str, encoding='latin1')

    # If this fails because of NaN-Entries, please remove all empty lines in this CSV by hand and try again.
    thursday_morn_df.dropna(how='all', inplace=True) # drop completely empty lines
    thursday_morn_df['timestamp'] = thursday_morn_df.timestamp.apply(calc_new_timestamp)


    # Besides the timestamp, the attack labels don't look too good either
    thursday_morn_df['label'] = thursday_morn_df['label'].str.replace('Web Attack .* Brute Force', 'BruteForce')
    thursday_morn_df['label'] = thursday_morn_df['label'].str.replace('Web Attack .* XSS', 'XSS')
    thursday_morn_df['label'] = thursday_morn_df['label'].str.replace('Web Attack .* Sql Injection', 'SQLInjection')

    # If everything went well, save changes to disk
    # As we've used a different datetime func, update the write to reflect the other timestamps
    thursday_morn_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_clean.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')


    # #### Afternoon
    thursday_aft_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_clean.csv'), dtype=str)
    thursday_aft_df['timestamp'] = thursday_aft_df.timestamp.apply(calc_new_timestamp)

    # If everything went well, save changes to disk
    thursday_aft_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_clean.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')


    print("Beginning conversion of Friday data")
    # ### Friday

    # #### Morning

    friday_morn_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Morning.pcap_ISCX_clean.csv'), dtype=str)
    friday_morn_df['timestamp'] = friday_morn_df.timestamp.apply(calc_new_timestamp)

    # If everything went well, save changes to disk
    friday_morn_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Morning.pcap_ISCX_clean.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')

    # #### Afternoon Portscan
    friday_aft_pscan_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_clean.csv'), dtype=str)
    friday_aft_pscan_df['timestamp'] = friday_aft_pscan_df.timestamp.apply(calc_new_timestamp)

    friday_aft_pscan_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_clean.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')


    # #### Afternoon DDoS

    # I don't know why, but pandas keep swallowing the flow_id column if I import the cleaned version.
    # So let's import the original, convert time and *then* fix the column headers
    friday_aft_ddos_df = pd.read_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'), dtype=str)

    # convert to 24h-format whilst converting to ISOdate
    friday_aft_ddos_df[' Timestamp'] = friday_aft_ddos_df[' Timestamp'].apply(calc_fri_timestamp)

    # If everything went well, save changes to disk
    # As the friday afternoon stuff has an additional CSV row, use the ddos_header (which includes the missing field "external_ip")
    friday_aft_ddos_df.to_csv(os.path.join(CIC_FOLDER_PATH, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_clean.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S', header=ddos_header)

    print("Finished CICIDS2017 cleanup!")
