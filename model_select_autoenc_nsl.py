import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD
    # Batch Size
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '128', '100', '0.2', '0.001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '256', '100', '0.2', '0.001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '1024', '100', '0.2', '0.001'])
    MLT.run.main(args)

    # # Epochs
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '10', '0.2', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '20', '0.2', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '50', '0.2', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '150', '0.2', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '200', '0.2', '0.001'])
    # MLT.run.main(args)

    # # Dropout
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.1', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.4', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.6', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.8', '0.001'])
    # MLT.run.main(args)

    # # Learning Rate
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.0001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.001'])
    # MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.01'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--nsl16', '--AutoEncoder', '512', '100', '0.2', '0.1', '--mail'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
