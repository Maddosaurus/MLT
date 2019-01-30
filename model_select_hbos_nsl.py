# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD

    # Defaults
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.1'])
    MLT.run.main(args)

    # n_bins
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '5', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '20', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '50', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '100', '0.1', '0.1'])
    MLT.run.main(args)


    # alpha
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.3', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.4', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.5', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.6', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.7', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.8', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.9', '0.1'])
    MLT.run.main(args)


    # tol
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.2'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.3'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.4'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.6'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.7'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.8'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--hbos', '10', '0.1', '0.9'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()