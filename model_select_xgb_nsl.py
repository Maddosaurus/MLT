# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD
    # estimators max_depth lr

    # max_depth
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '100', '3', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '100', '10', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '100', '50', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '100', '100', '0.1', '--mail'])
    # MLT.run.main(args)


    # estimators with max_depth = 10
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '50', '10', '0.1'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '100', '10', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '150', '10', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '200', '10', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--nsl16', '--XGBoost', '500', '10', '0.1'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()