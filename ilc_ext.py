from dataset import Dataset
from feature_map import FeatureMap, Layer
from constants import *
from utils import ressource_full_filename
import numpy as np
import os
import config
input_size = 30, 18, 18
output_size = 2

data_name = config.options.get("DATA").get("NAME", "")

def from_str_targets_to_numbers(targets):
    return [[1, -1] if target == 'elastic' else [-1, 1] for target in targets]
def from_numbers_to_str_targets(targets):
    return ["elastic" if target==[1, 0] else "inelastic" for target in targets]

def load_data(file, nbz=30, take=-1, take_start=0, method="old"):
    nbz  = input_size[0]
    file = ressource_full_filename(os.path.join(data_name, file), DATA)
    step = 30 / nbz
    
    if method=="old":
        data, targets = np.load(file)
    else:
        data = np.load(file)
        targets = data[:, -1]
        data = data[:, 0:-1]
    if take==-1:
        take=len(data)
    data = data[take_start:take+take_start]
    targets = targets[take_start:take+take_start]

    nb_examples = len(data)
    data = np.array(list(data)).reshape((nb_examples, 30, 18, 18))
    
    new_data = np.zeros((data.shape[0], nbz, data.shape[2], data.shape[3]))
    for i in xrange(0, nbz):    
        new_data[:, i, :, :] = np.mean(data[:, i*step:(i+1)*step, :, :], axis=1)
    
    #data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + (np.std(data, axis=0)==0))
    #data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + (np.max(data,axis=0)==np.min(data,axis=0)) )

    #data = binarize(data)
    #data = histo_equal(data)
    targets = from_str_targets_to_numbers(targets)
    targets = np.array((targets)).reshape((nb_examples, output_size))
    return data, targets


def load_ilc():

    whole = False
    nbz = 30
    if whole==True:
        data, targets = load_data("data.npy", nbz, take=-1, method="old")
        print data.shape, targets.shape
        ds = Dataset(data, targets)
        ds.shuffle()
        ds = Dataset(data, targets)
        #nb = 200
        #ds.truncate(nb) 
        train_ds, test_ds = ds.break_to_datasets((0.8, 0.2))
    else:
        data, targets = load_data("train.npy", nbz, take=-1, method="old")
        test_data, test_targets = load_data("test.npy", nbz, take=-1, method="old")
        train_ds = Dataset(data, targets)
        test_ds = Dataset(test_data, test_targets)
    return train_ds, test_ds


def build_3d_input_layer():
    input_layer_fmaps = []
    input_layer_fmaps.append(FeatureMap(input_size, name="input_layer"))
    return Layer(input_layer_fmaps) 

def build_output_layer(output_dim):
    return Layer([FeatureMap((output_size,), activation_func_name="tanh")])

def build_input_layer(input_dim):
    if input_dim == 2:
        input_layer = build_2d_input_layer()
    elif input_dim == 3:
        input_layer = build_3d_input_layer()
    else:
        raise Exception("wrong input dim")
    return input_layer

def build_2d_input_layer():
    input_layer_fmaps = []
    for z in xrange(30):
        input_layer_fmaps.append(FeatureMap((18, 18), name="input_layer%d" % (z,)))
    return Layer(input_layer_fmaps) 

def build_reduced_to_2d_input_layer(input_dim):
    return Layer([FeatureMap( (30, 18) )])

def build_hough_2d_input_layer(input_dim):
    return Layer([FeatureMap( (70, 180) )])


def to_2d(dataset, output):
    return dataset.sum(axis=2)
