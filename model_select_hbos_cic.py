import numpy
# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # Exhaustive search in full arg space

    parser = MLT.run.create_parser()

    for n_bins in range(1, 151):
        for alpha in numpy.arange(0.1, 1.0, 0.1):
            for tol in numpy.arange(0.1, 1.0, 0.1):
                arguments = [
                    '--unsupervised',
                    '-k', '10',
                    '--cic16', '--hbos',
                    '{:n}'.format(n_bins),
                    '{:2.1f}'.format(alpha),
                    '{:2.1f}'.format(tol)
                ]
                if n_bins == 150 and alpha == 0.9 and tol == 0.9:
                    # on the last run, send a mail
                    arguments.append('--mail')

                MLT.run.main(parser.parse_args(arguments))


if __name__ == '__main__':
    main()