import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Isolation Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--single', '--splunkr', '--IsolationForest', '350', '0.5', 'True'])
    MLT.run.main(args)

    # XGBOD
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--splunkr', '--XGBoost', '150', '50', '0.1'])
    MLT.run.main(args)

    #Random Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--splunkr', '--RandomForest', '10', '0'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
