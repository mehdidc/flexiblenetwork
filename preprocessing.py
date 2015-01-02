import numpy as np
from skimage import exposure
from skimage.transform import hough_line
from utils import linearize

def min_max_scaler(dataset, output):
    min_ = dataset.min(axis=0)
    max_ = dataset.max(axis=0)
    output[:] = (dataset - min_) / ( (max_ - min_) + (max_==min_))
    return output

def mean_std_scaler(dataset, output):
    mean = dataset.mean(axis=0)
    std = dataset.std(axis=0)
    output[:] = (dataset - mean) / (std + (std==0))
    return output

def histogram_equalization(dataset, output):
    for i in xrange(dataset.shape[0]):
        output[i] = exposure.equalize_hist(dataset[i])
    return output


def hough_transform_2d(dataset, output):
    nb_angles = 180
    max_distance =  2 * np.ceil(np.sqrt(dataset.shape[1] * dataset.shape[1] +
                                        dataset.shape[2] * dataset.shape[2]))
    result = np.zeros( (dataset.shape[0], max_distance, nb_angles), dtype=np.float32  )
    for i in xrange(dataset.shape[0]):
        out, _, _ = hough_line(dataset[i])
        result[i] = out.astype(np.float32)
    return result

def apply_preprocessing_chain(dataset, chain):
    for func in chain:
        dataset = func(dataset, output=dataset)
    return dataset

