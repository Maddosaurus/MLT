import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Isolation Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--nsl16', '--IsolationForest', '100', '0.4654', '0.1', 'True'])
    MLT.run.main(args)

    # HBOS
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--nsl16', '--HBOS', '2', '0.3', '0.2', '0.4654'])
    MLT.run.main(args)

    # AutoEncoder
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.4654', '0.1'])
    MLT.run.main(args)

    # XGBOD
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--nsl16', '--XGBoost', '150', '10', '0.1'])
    MLT.run.main(args)

    #Random Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--nsl16', '--RandomForest', '100', '0'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
