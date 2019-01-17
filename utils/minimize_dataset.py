# -*- coding: utf-8 -*-
import argparse
import os
from keras_wrapper.dataset import loadDataset, saveDataset


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
    # Load dataset
    ds = loadDataset(args.dataset)
    # Reinitialize values to empty
    ds.loaded_train = [False, False]
    ds.loaded_val = [False, False]
    ds.loaded_test = [False, False]

    ds.loaded_raw_train = [False, False]
    ds.loaded_raw_val = [False, False]
    ds.loaded_raw_test = [False, False]

    ds.len_train = 0
    ds.len_val = 0
    ds.len_test = 0
    # Remove data
    for key in ds.X_train.keys():
        ds.X_train[key] = None
    for key in ds.X_val.keys():
        ds.X_val[key] = None
    for key in ds.X_test.keys():
        ds.X_test[key] = None
    for key in ds.X_train.keys():
        ds.X_train[key] = None
    for key in ds.Y_train.keys():
        ds.Y_train[key] = None
    for key in ds.Y_val.keys():
        ds.Y_val[key] = None
    for key in ds.Y_test.keys():
        ds.Y_test[key] = None
    for key in ds.X_raw_train.keys():
        ds.X_raw_train[key] = None
    for key in ds.X_raw_val.keys():
        ds.X_raw_val[key] = None
    for key in ds.X_raw_test.keys():
        ds.X_raw_test[key] = None
    for key in ds.Y_raw_train.keys():
        ds.Y_raw_train[key] = None
    for key in ds.Y_raw_val.keys():
        ds.Y_raw_val[key] = None
    for key in ds.Y_raw_test.keys():
        ds.Y_raw_test[key] = None

    # Save dataset
    output_path = args.output if args.output else os.path.dirname(args.dataset)
    saveDataset(ds, output_path)
