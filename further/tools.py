
from feature_map import Learner
from error_rate import classification_error_rate, confusion_matrix
import sys
import numpy as np
from ILC import load_data
import gzip
from dataset import Dataset
if __name__ == "__main__":
    tool = sys.argv[1]

    if tool == "test":
        # arg1 : learner
        # arg 2 : npy file
        learner_file = sys.argv[2]
        npy_file = sys.argv[3]
        data, targets = load_data(npy_file)
        learner =  Learner.load(gzip.open(learner_file, "r"))
        ds = Dataset(data, targets)
        data, targets = learner.from_dataset_to_data(ds)
        output = learner.predict(data)
        output_layer = learner.layers[-1]
        output_fmap = output_layer.feature_maps[0]
        print classification_error_rate(output[output_fmap], targets[output_fmap])
        print confusion_matrix(output[output_fmap], targets[output_fmap])
