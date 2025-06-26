import os, sys
sys.path.append("./")
sys.path.append("../")

from config.params_parser.parser import ArgsParser



if __name__ == '__main__':
    args = ArgsParser().create_cv_config()
    print(args)

    args = ArgsParser().create_nlp_config()
    print(args)

    

    

