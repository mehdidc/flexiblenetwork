from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample


import numpy as np
from utils import ressource_full_filename, from_hamming, to_hamming
from dataset import Dataset

from constants import *

from dataset import Dataset
from feature_map import FeatureMap, Layer

import config
import os

data_name = config.options.get("DATA").get("NAME", "")

def load_data():

    train_filename = ressource_full_filename(os.path.join(data_name, "train.npy"), DATA)
    test_filename = ressource_full_filename(os.path.join(data_name, "test.npy"), DATA)
    (train_i, train_o), (test_i, test_o) = np.load(train_filename), np.load(test_filename)

    train_i, train_o = np.array(list(train_i)), np.array(list(train_o))
    test_i, test_o = np.array(list(test_i)), np.array(list(test_o))
    
    train_o = to_hamming(train_o)
    test_o = to_hamming(test_o)


    return Dataset(train_i, train_o), Dataset(test_i, test_o)

def load_data_whole():
    data_filename = ressource_full_filename(os.path.join(data_name, "data.npy"), DATA)
    data, targets = np.load(data_filename)
 
    data = np.array(list(data))
    
    targets = np.array(list(targets))

    targets = to_hamming(targets)
    #targets = targets[:, 0:10]

    print data.shape
    print targets.shape
    ds = Dataset(data, targets)
    #ds.shuffle()
    train_ds, test_ds = ds.break_to_datasets((0.8, 0.2))
    return train_ds, test_ds

def load_data_for_testing():
    data_filename = ressource_full_filename(os.path.join(data_name, "predict.npy"), DATA)
    data = np.load(data_filename)
    print len(data)
    return Dataset(data, None, check=False), Dataset(data, None, check=False)

