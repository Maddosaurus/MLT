import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD
    # Batch Size
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '1024', '100', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '2048', '100', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.1'])
    MLT.run.main(args)

    # Epochs
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '10', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '20', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '50', '0.2', '0.1'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '150', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '200', '0.2', '0.1'])
    MLT.run.main(args)

    # Dropout
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.1', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.4', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.6', '0.1'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.8', '0.1'])
    MLT.run.main(args)

    # Contamination
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.2'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.3'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.4'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--cic16', '--AutoEncoder', '4096', '100', '0.2', '0.5', '--mail'])
    MLT.run.main(args)


if __name__ == '__main__':
    main()
