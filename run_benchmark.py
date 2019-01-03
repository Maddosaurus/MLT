# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Run the final original and optimized benchmarks.

    #Random Forest
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--nsl', '--rf', '1000', '50'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--cic6s', '--rf', '50', '100'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--cic6r', '--rf', '500', '100'])
    MLT.run.main(args)


    # XGBoost
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--nsl', '--xgb', '1000', '10', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--cic6s', '--xgb', '10', '10', '0.01'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--single', '--cic6r', '--xgb', '500', '10', '0.01'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
