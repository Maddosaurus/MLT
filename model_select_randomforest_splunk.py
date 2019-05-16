# Fix the module import path for usage without module installation
import os
import sys
sys.path.insert(0, os.path.abspath('./MLT'))

import MLT.run

def main():
    # NSL_KDD
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 10', '0'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 20', '0'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 50', '0'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '100', '0'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '200', '0'])
    MLT.run.main(args)
    
    ####
    ####
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 10', '10'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 20', '10'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 50', '10'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '100', '10'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '200', '10'])
    MLT.run.main(args)

    ####
    ####
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 10', '100'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 20', '100'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', ' 50', '100'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '100', '100'])
    MLT.run.main(args)
    
    parser = MLT.run.create_parser()
    args = parser.parse_args(['-k', '10', '--splunkr', '--RandomForest', '200', '100'])
    MLT.run.main(args)



if __name__ == '__main__':
    main()