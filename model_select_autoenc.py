import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD

    # Defaults
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--LSTM2', '32', '100', '0.01'])
    MLT.run.main(args)
