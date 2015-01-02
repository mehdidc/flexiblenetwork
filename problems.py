import sys

from feature_map import FeatureMap, Learner, Layer
from connections import ConnectorConvolution, ConnectorAll, ConvolutionalWeightMatrix, ConnectorMaxpooling, ConnectorAveragePooling
import json

import numpy as np
from data import get_mnist

from error_rate import classification_error_rate, where_error, confusion_matrix
from easy import stochastic_training, build_weighted_pool_layer, build_convpool_layers
from collections import defaultdict
import config
from show import show_3d

def ILC_E():

    real_test_dataset, real_test_targets = np.load("test.npy")

    def load_data(file):
        return np.load(file)
    
    input_fmap = FeatureMap((7,))
    input_layer = Layer([input_fmap])

    output_fmap = FeatureMap((2,), activation_func_name="tanh")
    output_layer = Layer([output_fmap])

    def get_images_and_labels(data):
        features = np.array([d[0:-1] for d in data]).astype('float')
        targets = np.array([ ([1, 0] if d[-1]=="elastic" else [0,1]) for d in data])
        return {input_fmap:features}, {output_fmap:targets}
    train = load_data("train-E.npy")
    
    train_images, train_labels = get_images_and_labels(train)

    
    fc_fmap = FeatureMap((50,), activation_func_name="tanh")
    fc_layer = Layer([fc_fmap])

    # Connect Feature maps:
    # Input (28x28) ---> Conv1(5, 5) ---> 24x24 --> Pool1(2, 2) --> 12x12
    #conv1, pool1 = build_convpool_layers(input_layer, (11, 11), (2, 2), nbconv=1, layer=1, all_all=True)
    # 12x12 ---> Conv2(5, 5) --> 8x8 ---> pool2(2, 2) ---> 4x4
    #conv2, pool2 = build_convpool_layers(pool1, (3, 4), (2, 2), nbconv=1, layer=2, all_all=True)

    #for fmap in pool1.feature_maps:
#       ConnectorAll(fmap, fc_fmap).connect()

    ConnectorAll(input_fmap, fc_fmap).connect()

    ConnectorAll(fc_fmap, output_fmap).connect()
    #learner = Learner([input_layer, conv1, pool1, fc_layer, output_layer])
    learner = Learner([input_layer, fc_layer, output_layer])
    learner.alpha = 0.001
    print "training"
    batch_size = 50

    data = load_data("test-E.npy")
    test_images, test_labels = get_images_and_labels(data)

    try:
        for i in xrange(2000):
            print learner.train_batches(train_images, train_labels, batch_size=batch_size, it=i)

            if i % 10 == 0:
                print
                out  = learner.predict(train_images)
                error_rate = classification_error_rate(train_labels[output_fmap], out[output_fmap])
                print "train error rate : %f" % (error_rate,)
                m = confusion_matrix(train_labels[output_fmap], out[output_fmap])
                print "train confusion matrix : "
                print m

                out  = learner.predict(test_images)
                error_rate = classification_error_rate(test_labels[output_fmap], out[output_fmap])
                print "test error rate : %f" % (error_rate,)
                m = confusion_matrix(test_labels[output_fmap], out[output_fmap])
                print "test confusion matrix : "
                print m
                print
                
    except KeyboardInterrupt:
        print "interrupted..."

    out = learner.predict(test_images)
    where = list(where_error(test_labels[output_fmap], out[output_fmap]))
    show_3d(real_test_dataset[where], real_test_targets[where])


def ILC_2D():

    real_test_dataset, real_test_targets = np.load("test.npy")

    def load_data(file):
        return json.loads(open(file, "r").read())
    
    input_fmap = FeatureMap((18, 18))
    input_layer = Layer([input_fmap])

    output_fmap = FeatureMap((2,))
    output_layer = Layer([output_fmap])

    def get_images_and_labels(data):
        images = [[ i[0] for i in d[0:-1]] for d in data]
        labels = [ [1, 0] if d[-1] == 1 else [0, 1] for d in data]
        return {input_fmap:np.array(images).reshape((len(images), 18, 18))}, {output_fmap:np.array(labels)}
    train = load_data("train.json")
    
    train_images, train_labels = get_images_and_labels(train)

    
    fc_fmap = FeatureMap((10,), activation_func_name="relu")
    fc_layer = Layer([fc_fmap])

    # Connect Feature maps:
    # Input (28x28) ---> Conv1(5, 5) ---> 24x24 --> Pool1(2, 2) --> 12x12
    #conv1, pool1 = build_convpool_layers(input_layer, (11, 11), (2, 2), nbconv=1, layer=1, all_all=True)
    # 12x12 ---> Conv2(5, 5) --> 8x8 ---> pool2(2, 2) ---> 4x4
    #conv2, pool2 = build_convpool_layers(pool1, (3, 4), (2, 2), nbconv=1, layer=2, all_all=True)

    #for fmap in pool1.feature_maps:
#       ConnectorAll(fmap, fc_fmap).connect()

    ConnectorAll(input_fmap, fc_fmap).connect()

    ConnectorAll(fc_fmap, output_fmap).connect()
    #learner = Learner([input_layer, conv1, pool1, fc_layer, output_layer])
    learner = Learner([input_layer, fc_layer, output_layer])
    print "training"
    batch_size = 10

    data = load_data("test.json")
    test_images, test_labels = get_images_and_labels(data)

    try:
        for i in xrange(2000):
            print learner.train_batches(train_images, train_labels, batch_size=batch_size, it=i)

            if i % 10 == 0:
                print
                out  = learner.predict(train_images)
                error_rate = classification_error_rate(train_labels[output_fmap], out[output_fmap])
                print "train error rate : %f" % (error_rate,)
                m = confusion_matrix(train_labels[output_fmap], out[output_fmap])
                print "train confusion matrix : "
                print m

                out  = learner.predict(test_images)
                error_rate = classification_error_rate(test_labels[output_fmap], out[output_fmap])
                print "test error rate : %f" % (error_rate,)
                m = confusion_matrix(test_labels[output_fmap], out[output_fmap])
                print "test confusion matrix : "
                print m
                print
                
    except KeyboardInterrupt:
        print "interrupted..."

    out = learner.predict(test_images)
    where = list(where_error(test_labels[output_fmap], out[output_fmap]))
    show_3d(real_test_dataset[where], real_test_targets[where])


def xor():
    input_layer = FeatureMap((2,))
    hidden_layer = FeatureMap((2,))
    output_layer = FeatureMap((1,), name="output_layer")

    ConnectorAll(input_layer, hidden_layer).connect()
    ConnectorAll(hidden_layer, output_layer).connect()

    layers = [Layer([input_layer]), Layer([hidden_layer]), Layer([output_layer])]
    learner = Learner(layers, options={"gradient_check": False})

    X = {input_layer: np.array([[0, 0], [0, 1], [1, 1], [1, 0]])}
    y = {output_layer: np.array([ [0], [1], [0], [1] ])}
    
    for it in xrange(1000):
        print learner.train(X, y)

    print learner.predict(X)


def conv():
    nb = 10
    x, y = 10, 10
    cx1, cy1 = 5, 5
    px1, py1 = 2, 2
    cx2, cy2 = 5, 5
    px2, py2 = 2, 2
    two = True
    nbc = 10
    fc = 10
    nb1, nb2 = 1, 1

    #data = np.sin(np.random.random((nb, y, x)) * 10)

    #data = np.random.randint(0, 256, size=(nb, y, x)) 

    data, target, dimensions =  get_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", take=nb)
    data = np.array(data)
    x, y = dimensions
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0))
    #data /= 255.
    data = data.reshape((nb, y, x))
    

    def T(r):
        #r = np.random.randint(0, nbc)
        l = [0] * nbc
        l[r] = 1
        return l
    target = np.array([T(target[i]) for i in xrange(nb)])

    hidden = []

    input_fmap = FeatureMap((y, x), name="input")
    output_fmap = FeatureMap((nbc,), activation_func_name="id", name="out")
    output_layer = Layer([output_fmap])
    input_layer = Layer([input_fmap])
    """
    # Connect Feature maps:
    # Input (28x28) ---> Conv1(5, 5) ---> 24x24 --> Pool1(2, 2) --> 12x12
    conv1, pool1 = build_convpool_layers(input_layer, (cy1, cx1), (py1, px1), nbconv=nb1, layer=1)
    hidden.append(conv1)
    hidden.append(pool1)
    
    # 12x12 ---> Conv2(5, 5) --> 8x8 ---> pool2(2, 2) ---> 4x4
    if two:
        conv2, pool2 = build_convpool_layers(pool1, (cy2, cx2), (py2, px2), nbconv=nb2, layer=2, all_all=True)
        hidden.append(conv2)
        hidden.append(pool2)
        last = pool2
    else:
        last = pool1
    """

    """
    conv1 = FeatureMap((28-5+1, 28-5+1))
    pool1 = FeatureMap((12, 12))

    L = Layer([conv1])
    hidden.append(L)
    L = Layer([pool1])
    hidden.append(L)

    ConnectorConvolution((5, 5), input_fmap, conv1).connect()
    ConnectorMaxpooling((2, 2), conv1, pool1).connect()


    last = L
    """
    conv = FeatureMap((28-5+1, 28-5+1), activation_func_name="relu")
    convl = Layer([conv])
    hidden.append(convl)
    ConnectorConvolution((5, 5), input_fmap, conv).connect()
    
    #fcX = FeatureMap((4, 4))
    #fcXl = Layer([fcX])
    #hidden.append(fcXl)
    #ConnectorAll(input_fmap, fcX).connect()

    apool = FeatureMap((12, 12), activation_func_name="id")
    ConnectorAveragePooling(2, (2, 2), conv, apool).connect()
    apool_l = Layer([apool])
    hidden.append(apool_l)
    last = apool_l

    """
    avg = FeatureMap((7, 7),name="avg")
    avgl = Layer([avg])
    ConnectorAveragePooling(2, (4, 4), input_fmap, avg).connect()
    last = avgl
    hidden.append(avgl)
    """
    #last = input_layer
    
    # pas toucher
    fc_fmap = FeatureMap((fc,), name="fc", activation_func_name="relu")
    fc_layer = Layer([fc_fmap])
    hidden.append(fc_layer)

    for fmap in last.feature_maps:
        ConnectorAll(fmap, fc_fmap).connect()
    ConnectorAll(fc_fmap, output_fmap).connect()

    data = {input_fmap: data}
    target = {output_fmap: target}
    
    layers = [input_layer] + hidden + [output_layer]
    learner = Learner(layers, options={"gradient_check": False, "loss_function": "softmax"})
    """
    for it in xrange(1000):
        print it
        print learner.train_batches(data, target, batch_size=10)
        out = learner.predict(data)
        if it % 10 == 0:
            print classification_error_rate(target[output_fmap], out[output_fmap])
    """
    ds = Dataset(data[input_fmap], target[output_fmap])
    stochastic_training(
        learner,
        ds, ds,
        batch_size=10,
        monitor_size=100,
        parallel_batch_size=1,
        nb_iterations=10000
    )
    

from dataset import Dataset
def mnist():
    images, labels, dimensions =  get_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", take=100)
    nrow, ncol = dimensions

    ds = Dataset(images, labels)
    ds.shuffle()
    
    ds.transform_input_whole(lambda x: (x / 255.))
    
    ds.transform_input_whole(lambda x: x.reshape(x.shape[0], nrow, ncol))
    ds.transform_output_from_classid_to_binvectors()
    

    # Divide to train & test
    train_ratio = 0.8
    test_ratio = 1. - train_ratio

    train_ds, test_ds = ds.break_to_datasets([train_ratio, test_ratio])
    

    # Create the neural network architecture :

    # Create feature maps
    input_fmap = FeatureMap((28, 28), name="input_layer%d" % (0,))
    input_layer = Layer([input_fmap])
    output_fmap = FeatureMap((10,), name="output_layer", activation_func_name="tanh")
    fc_fmap = FeatureMap((10,), activation_func_name="tanh")
    fc_layer = Layer([fc_fmap])
    output_layer = Layer([output_fmap]) 

    # Connect Feature maps:
    # Input (28x28) ---> Conv1(5, 5) ---> 24x24 --> Pool1(2, 2) --> 12x12
    conv1, pool1 = build_convpool_layers(input_layer, (5, 5), (1, 1), (2, 2), nbconv=2, layer=1, all_all=True)
    # 12x12 ---> Conv2(5, 5) --> 8x8 ---> pool2(2, 2) ---> 4x4
    conv2, pool2 = build_convpool_layers(pool1, (5, 5), (1, 1), (2, 2), nbconv=4, layer=2, all_all=True)
    # 4x4 ---> FullyConnected(50)
    # FullyConnected(50) --> OutputLayer(10)
    for fmap in pool2.feature_maps:
        ConnectorAll(fmap, fc_fmap).connect()
    ConnectorAll(fc_fmap, output_fmap).connect()
    learner = Learner([input_layer, conv1, pool1,  conv2, pool2, fc_layer, output_layer],
                      options={"gradient_check": False, "loss_function": "mse"})
    learner.alpha = 0.01
    test_data_i, test_data_o = learner.from_dataset_to_data(test_ds)
    for i in xrange(200):
        print "iteration : %d" % (i,)
        data_i, data_o = learner.from_dataset_to_data(train_ds)
        print learner.train_batches(data_i, data_o, batch_size=20, it=i)
        if i % 1 == 0:
            print
            out = learner.predict(data_i)
            error_rate = classification_error_rate(data_o[output_fmap], out[output_fmap])
            print "Overall train error rate : %f" % (error_rate,)
            m = confusion_matrix(data_o[output_fmap], out[output_fmap])
            print "train confusion matrix : "
            print m

            out = learner.predict(test_data_i)
            error_rate = classification_error_rate(test_data_o[output_fmap], out[output_fmap])
            print "Overall test error rate : %f" % (error_rate,)
            m = confusion_matrix(test_data_o[output_fmap], out[output_fmap])
            print "test confusion matrix : "
            print m
            print
    
    
    if config.options["PARALLELISM"]==True:
        learner.workers.close()

    test_images, test_labels = learner.from_dataset_to_data(test_ds)
    out =  learner.predict(test_images)
    error_Rate = classification_error_rate(test_labels[output_fmap], out[output_fmap])
    print "Overall test error rate : %f" % (error_rate,)
    return learner

from utils import whiten

from scipy.sparse import lil_matrix 

if __name__ == "__main__":
    import time
    s = int(time.time())
    print "seed : %d"  % (s,)   
    np.random.seed(s)
    learner = ILC()


class ConnectorWeightedPooling(ConnectorKernel):
    
    def __init__(self, *args, **kwargs):
        super(ConnectorWeightedPooling, self).__init__(*args, **kwargs)
        assert np.array_equal(np.array(self.to_fmap.dimensions) * np.array(self.kernel_size), 
                              np.array(self.from_fmap.dimensions))

    def connect(self):
        W = WeightedPoolingMatrix(self.kernel_size)
        self.to_fmap.M[self.from_fmap] = W


