import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Batch Size
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '1024', '100', '0.2', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '2048', '100', '0.2', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.001', '--mail']) # <-- Winner
    # MLT.run.main(args)

    # # Epochs with Batch = 4096
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '50', '0.2', '0.195', '0.001']) # NO clear winner, so Epoch=100
    # MLT.run.main(args)
    
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '150', '0.2', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '200', '0.2', '0.195', '0.001', '--mail'])
    # MLT.run.main(args)

    # # Dropout with Batch=4096, Epoch=100
    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.1', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.001']) # <-- Winner? 
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.4', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.6', '0.195', '0.001'])
    # MLT.run.main(args)

    # parser = MLT.run.create_parser()
    # args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.8', '0.195', '0.001'])
    # MLT.run.main(args)

    # # Learning Rate
    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.0001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.001'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.01'])
    MLT.run.main(args)

    parser = MLT.run.create_parser()
    args = parser.parse_args(['--unsupervised', '-k', '10', '--CICt', '--AutoEncoder', '4096', '100', '0.2', '0.195', '0.1', '--mail'])
    MLT.run.main(args)

if __name__ == '__main__':
    main()
