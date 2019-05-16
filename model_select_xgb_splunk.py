# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # estimators max_depth lr

    # max_depth
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '100', '3', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '100', '10', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '100', '50', '0.1']) # <-- Winner
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '100', '100', '0.1', '--mail'])
    # MLT.run.main(args)


    # estimators with max_depth = 50
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '50', '50', '0.1'])
    # MLT.run.main(args)
    
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '100', '50', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '150', '50', '0.1']) # <--- Winner!
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '200', '50', '0.1'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '500', '50', '0.1', '--mail'])
    # MLT.run.main(args)


    # learn rate with max_depth = 50 and n_estimators = 150 
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '150', '50', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '150', '50', '0.01'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '150', '50', '0.001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--XGBoost', '150', '50', '0.0001', '--mail'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()