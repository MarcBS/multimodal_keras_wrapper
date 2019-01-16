# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import logging
import ast
from keras_wrapper.dataset import loadDataset, saveDataset

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Minimizes a dataset by removing the data stored in it: Tranining, development and test. "
                                     "The rest of parameters are kept."
                                     "Useful for reloading datasets with new data.")
    parser.add_argument("-d", "--dataset", required=True, help="Stored instance of the dataset")
    parser.add_argument("-o", "--output", help="Output dataset file.",
                        default="")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    ds = loadDataset(args.dataset)
    # Reinitialize values to empty sets
    ds.loaded_train = [False, False]
    ds.loaded_val = [False, False]
    ds.loaded_test = [False, False]
    ds.len_train = 0
    ds.len_val = 0
    ds.len_test = 0
    ds.X_train = dict()
    ds.X_val = dict()
    ds.X_test = dict()
    ds.Y_train = dict()
    ds.Y_val = dict()
    ds.Y_test = dict()
    ds.loaded_raw_train = [False, False]
    ds.loaded_raw_val = [False, False]
    ds.loaded_raw_test = [False, False]
    ds.X_raw_train = dict()
    ds.X_raw_val = dict()
    ds.X_raw_test = dict()
    ds.Y_raw_train = dict()
    ds.Y_raw_val = dict()
    ds.Y_raw_test = dict()
    output_path = args.output if args.output is not None else args.dataset
    saveDataset(ds, output_path)