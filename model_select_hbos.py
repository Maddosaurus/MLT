# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Defaults
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.1', '0.1'])
    MLT.run.main(args)

    # n_bins
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '5', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '20', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '50', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '100', '0.1', '0.1'])
    MLT.run.main(args)


    # alpha
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.5', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.8', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '1.0', '0.1'])
    MLT.run.main(args)


    # tol
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.1', '0.2'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.1', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.1', '0.8'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '--kfolds', '10', '--nsl', '--hbos', '10', '0.1', '1.0'])
    MLT.run.main(args)