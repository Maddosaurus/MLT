#!/usr/bin/env python3
import argparse
import heapq
import json
import natsort
import os
import pandas
from pandas.io.json import json_normalize

def main():
    parser = argparse.ArgumentParser("Find top n measurements of every metric ")
    parser.add_argument('--path', '-p', required=True, help='Path where to find all measurements')
    parser.add_argument('--model', '-m', required=True, help='Name of the model for all measurements')
    parser.add_argument('--n', '-n', required=True, help='Number of top elements to show')
    args = parser.parse_args()

    find_top_n(args.path, args.model, args.n)


def find_top_n(path, model, n):
    n = int(n)
    folderlist = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    datadf = pandas.DataFrame()
    
    print('Reading data')
    for folder in folderlist:
        # Get the data itself
        with open (os.path.join(path, folder, model+"_metrics.json")) as fmetric:
            jdata = json.load(fmetric)
            ndata = json_normalize(jdata)
            ndata = ndata.filter([
                'precision.mean', 'precision.sd',
                'recall.mean', 'recall.sd',
                'f1_score.mean', 'f1_score.sd',
                'auc.mean', 'auc.sd',
                'training_time_mean'
            ])

        # as well as the call params
        try:
            with open(os.path.join(path, folder, 'call_parameters.txt')) as cparms:
                call_vals = None
                for line in cparms:
                    try:
                        call_vals = line.split('[')[1].split(']')[0].replace("'", "")
                        call_vals = call_vals.split(', ')
                        call_df = pandas.DataFrame(data=[call_vals])
                        call_df['folder_name'] = folder
                        ndata = pandas.concat([ndata, call_df], axis=1)
                        datadf = datadf.append(ndata, ignore_index=True, sort=False)
                    except IndexError:
                        pass # TODO: Find a more efficient alternative
        except FileNotFoundError:
            pass

    print("Top {} by F1 score:".format(n))
    _print_as_ltx_table(datadf.nlargest(n, 'f1_score.mean'))

    print("Top {} by Precision:".format(n))
    _print_as_ltx_table(datadf.nlargest(n, 'precision.mean'))

    print("Top {} by Recall:".format(n))
    _print_as_ltx_table(datadf.nlargest(n, 'recall.mean'))

    print("Top {} by AUC:".format(n))
    _print_as_ltx_table(datadf.nlargest(n, 'auc.mean'))


def _print_as_ltx_table(pandasDF):
    print("Layout: row 0 & row 1 & row 2 & row3 & Prec & Recall & F1 pm F1.sd & auc & runtime_mean & folder_name")
    
    for index, row in pandasDF.iterrows():
        print("{} & {} & {} & {} & {:4.2f} & {:4.2f} & {:4.2f} $\pm$ {:4.2f} & {:4.2f} & {} & {}\\\\".format(
            row[0],
            row[1],
            row[2],
            row[3],
            float(row['precision.mean'])*100,
            float(row['recall.mean'])*100,
            float(row['f1_score.mean'])*100,
            float(row['f1_score.sd'])*100,
            float(row['auc.mean'])*100,
            row['training_time_mean'],
            row['folder_name'],
        ))

if __name__ == '__main__':
    main()
