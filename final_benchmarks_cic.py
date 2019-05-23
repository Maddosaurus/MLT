import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Isolation Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--CICt', '--IsolationForest', '100', '0.5', 'True'])
    MLT.run.main(args)

    # HBOS
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--CICt', '--HBOS', '5', '0.2', '0.1'])
    MLT.run.main(args)

    # AutoEncoder
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.001'])
    MLT.run.main(args)

    # XGBOD
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--CICt', '--XGBoost', '100', '10', '0.1'])
    MLT.run.main(args)

    #Random Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--CICt', '--RandomForest', '100', '0', '--mail'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
