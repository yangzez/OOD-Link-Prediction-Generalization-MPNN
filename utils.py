import sys
import logging
import numpy as np


def transform_labels(y, n_classes):
    y_new = np.zeros((y.shape[0], n_classes), dtype=np.int32)
    for i in range(y.shape[0]):
        y_new[i, y[i]] = 1
    return y_new


def accuracy(gold, pred):
    try:
        denom = gold.shape[0]
        nom = (gold.squeeze().long() == pred).sum()
        ret = float(nom) / denom
    except:
        denom = gold.data.shape[0]
        nom = (gold.data.squeeze().long() == pred.data).sum()
        ret = float(nom) / denom
    return ret


def bin_config(get_arg_func):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # set logger
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
    try:
        # if output_folder is specified in the arguments
        # put the log in there
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
        fpath = os.path.join(args.output_folder, 'log')
    except:
        # otherwise, create a log file locally
        fpath = 'log'
    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return args
