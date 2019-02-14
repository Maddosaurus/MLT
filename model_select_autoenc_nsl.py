import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD

    # Defaults
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '32', '100', '0.2', '0.5'])
    MLT.run.main(args)

    # Batch Size
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '64', '100', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '256', '100', '0.2', '0.5'])
    MLT.run.main(args)

    # Epochs
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '10', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '20', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '50', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '150', '0.2', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '200', '0.2', '0.5'])
    MLT.run.main(args)

    # Dropout
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.1', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.4', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.6', '0.5'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.8', '0.5'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()
