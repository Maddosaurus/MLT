# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # CIC16 Randomized

    # Estimators
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '10', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '20', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '40', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '60', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '80', '1.0', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '1.0', 'True'])
    MLT.run.main(args)


    # max_features
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.1', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.2', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.3', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.4', 'True'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.5', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.6', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.7', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.8', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '0.9', 'True'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '1.0', 'True'])
    MLT.run.main(args)


    # Bootstrap
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--IsolationForest', '100', '1.0', 'False', '--mail'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()