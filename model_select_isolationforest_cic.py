# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # CIC16 Randomized

    # Defaults
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '1.0', 'True'])
    MLT.run.main(args)

    # Estimators
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '10', '0.1', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '20', '0.1', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '40', '0.1', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '60', '0.1', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '80', '0.1', '1.0', 'True'])
    MLT.run.main(args)


    # Contamination
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.2', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.3', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.4', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.5', '1.0', 'True'])
    MLT.run.main(args)


    # max_features
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '0.2', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '0.4', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '0.6', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '0.8', 'True'])
    MLT.run.main(args)


    # Bootstrap
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--IsolationForest', '100', '0.1', '1.0', 'False', '--mail'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()