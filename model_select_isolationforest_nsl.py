# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD

    # Estimators
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '100', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '150', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '200', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '250', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '300', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '350', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '450', '1.0', 'True'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '500', '1.0', 'True', '--mail'])
    # MLT.run.main(args)


    # max_features
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '100', '0.1', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.2', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.3', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.4', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.5', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.6', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.7', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.8', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '0.9', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '1.0', 'True'])
    MLT.run.main(args)


    # Bootstrap
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--IsolationForest', '400', '1.0', 'False', '--mail'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()