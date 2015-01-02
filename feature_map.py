

import signal   
import sys
import numpy as np

from itertools import izip
from collections import OrderedDict

from connections import ConnectorAll, ConnectorConvolution, ConnectorMaxpooling, ConnectorWeightedPooling
from connections import ConvolutionalWeightMatrix

from data import get_mnist, list_with_one_in

from utils import linearize

import json
from collections import defaultdict

import random

import config
from multiprocessing import Pool
from itertools import repeat
from utils import Workers, get_batches, histogramEqualization

from utils import get_max_rel_error, safe_log, safe_exp
from utils import lumberstack, translate


sigmoid = lambda x:1/(1+safe_exp(-x))
d_sigmoid = lambda x : (sigmoid(x) * (1 - sigmoid(x)))

shapes = {}

def nrelu(x):
    if x.shape in shapes:
        return shapes[x.shape]
    else:
        shapes[x.shape] = np.random.normal(0, 1, size=x.shape)

    return np.maximum(0, x + shapes[x.shape])

def nrelu_d(x):
    if x.shape in shapes:
        return shapes[x.shape]
    else:
        shapes[x.shape] = np.random.normal(0, 1, size=x.shape)

    return ((x + shapes[x.shape]) > 0) * 1.



activation_funcs = {
    "sigmoid": (sigmoid, d_sigmoid),
    "id" : (lambda x:x, np.vectorize(lambda x:1)),
    #"relu": (lambda x:safe_log(1 + safe_exp(x)), lambda x:1/(1+safe_exp(-x)))
    "relu": (np.vectorize(lambda x: max(0, x)), np.vectorize(lambda x:1 if x > 0 else 0)),

    "nrelu": (nrelu, nrelu_d),

    #"relu" : (lambda x:np.maximum(x + np.random.normal(0, np.std(x), size=x.shape), 0), 
    #         lambda x:((x + np.random.normal(0, np.std(x), size=x.shape))>0).astype('float')),
    "tanh": (np.tanh, lambda x:(1-np.tanh(x)**2)),
    "tanhh": (lambda x : 1.7159 * np.tanh(2.*x/3.), lambda x:(1.7159*2./3* (1-np.tanh(2.*x/3.)**2)  ) )
}

aggreg_funcs = {
    "concat": lambda r: np.concatenate( map(lambda a:a[:, np.newaxis], r), axis=1),
    "sum": lambda r: reduce(lambda x, y: x + y, r[1:], r[0])
}

def softmax_loss(real, predicted, weights):
    where_max = np.argmax(real, axis=1)
    Ai = np.choose(where_max, predicted.T)
    Amax = np.max(predicted, axis=1)
    Amin = np.min(predicted, axis=1)
    A = predicted
    return -np.sum(safe_log(safe_exp(Ai) / np.sum(safe_exp(A), axis=1)))

def softmax_delta(real, predicted, E, activation, d_activation, weights):
    nb_examples = real.shape[0]

    where_max = np.argmax(real, axis=1)
    Amax = np.max(predicted, axis=1)[:, np.newaxis]
    Amin = np.min(predicted, axis=1)[:, np.newaxis]
    Ai = np.choose(where_max, predicted.T)

    A = predicted
    
    # substract A maxes to avoid overflows

    sum_A = np.sum(safe_exp(A), axis=1).reshape(nb_examples, 1)
    P = safe_exp(A) / sum_A


    delta = -P
    for i in xrange(nb_examples):
        delta[i][where_max[i]] += 1

    return -delta

loss = {
    "mse": (lambda real, predicted, weights: (np.sum(   (( (predicted - real)) ** 2))   ), # Compute the loss
            lambda real, predicted, E, activation, d_activation, weights: ( # Compute Delta
                (2 * d_activation(E) * ( (predicted - real))))),
    "softmax":  (softmax_loss, softmax_delta),
}


from uuid import uuid1
class FeatureMap(object):

    def __init__(self, dimensions, activation_func_name="sigmoid", name=""):
        self.dimensions = dimensions
        self.features = None
        self.activation_func_name = activation_func_name
        self.M = {}
        self.name = name
        self.id = uuid1()
        self.W = None

        self.aggreg = "sum"

    def do(self, operation):
        for k, W in self.M.items():
            self.M[k] = operation(W)
        return self
    
    def do_with(self, other, operation):
        for (k, W), (k_other, W_other) in zip(self.M.items(), other.M.items()):
            self.M[k] = operation(W, W_other)
        return self

    def init_weights(self, range_val=None):
        for W in self.M.values():
            W.init()

    def activation(self, x):
        a, _ = activation_funcs[self.activation_func_name]
        return a(x)

    def d_activation(self, x):
        _, a = activation_funcs[self.activation_func_name]
        return a(x)

    def get_nb_features(self):
        return np.prod(self.dimensions)

    def transform(self, X, from_map):
        if len(self.M) == 0: # (input layer)
            return X
        return self.M[from_map].transform(X)

    def activate(self, X):
        return self.activation(X)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class Layer(object):

    def __init__(self, feature_maps):
        self.feature_maps = feature_maps

    def transform(self, X):
        results = {}
        for feature_map in self.feature_maps:
            nb = 1.
            r = []
            for from_fmap in feature_map.M.keys():
                result = feature_map.transform(X[from_fmap], from_fmap)
                r.append(result)
            results[feature_map] = aggreg_funcs[feature_map.aggreg](r)
            #results[feature_map] /= nb
        return results

    def activate(self, X):
        R = {}
        for feature_map in self.feature_maps:
            R[feature_map] = feature_map.activate(X[feature_map])   
        return R

    def init_weights(self, range_val=None):
        for feature_map in self.feature_maps:
            feature_map.init_weights(range_val)
    
    def do(self, operation):
        for i, feature_map in enumerate(self.feature_maps):
            self.feature_maps[i] = feature_map.do(operation)
        return self

    def do_with(self, layer, operation):
        for i, (feature_map, feature_map_other) in enumerate(zip(self.feature_maps, layer.feature_maps)):
            self.feature_maps[i] = feature_map.do_with(feature_map_other, operation)
        return self

import cPickle as pickle

import copy
class Learner(object):

    ALPHA_DEFAULT = 0.001
    MOMENTUM_DEFAULT = 0.5
    LAMBDA1_DEFAULT = 0.
    LAMBDA2_DEFAULT = 0.
    BETA_DEFAULT = 0.1

    EPSILON = 1e-5
    def __init__(self, layers, options=None):
        self.layers = layers
        self.workers = None

        self.old_gradients = [None] * len(self.layers)
        self.velocity_W = [None] * len(self.layers)
        self.velocity_b = [None] * len(self.layers)
        
        self.options = defaultdict(bool)
        self.options["loss_function"] = "mse"
    
        if options is not None:
            self.options.update(options)

        self.alpha = self.options.get("alpha", Learner.ALPHA_DEFAULT)
        self.momentum = self.options.get("momentum", Learner.MOMENTUM_DEFAULT)
        self.lambda1 = self.options.get("lambda1", Learner.LAMBDA1_DEFAULT)
        self.lambda2 = self.options.get("lambda2", Learner.LAMBDA2_DEFAULT)
        self.beta = self.options.get("beta", Learner.BETA_DEFAULT)
        self.class_weights = self.options.get("class_weights", None)

        self.init_parallelism()

    def init_parallelism(self):
        if config.options["PARALLELISM"]==True and self.options["lightweight"] == False:
            def get_gradients_directly(learner, X, y):
                return learner.get_gradients_directly(X, y)
            def get_gradients(learner, X, y, nb_batches):
                learner.train_batches(X, y, nb_batches)
                return learner
            
            self.workers = Workers(get_gradients, size=config.options["PARALLEL_POWER"])
    
    def do(self, operation):
        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.do(operation)
        return self.layers
    
    def do_with(self, learner, operation):
        for i, (layer, layer_other) in enumerate(zip(self.layers, learner.layers)):
            self.layers[i] = layer.do_with(layer_other, operation)

    def save(self, fd):
        x = self.workers
        self.workers = None
        pickle.dump(self, fd)
        self.workers = x
    
    def init(self):
        for layer in self.layers:
            layer.init_weights()
        self.old_gradients = [None]  * len(self.layers)

    @staticmethod
    def load(fd):
        learner = pickle.load(fd)
        learner.init_parallelism()
        return learner

    def get_lightweight_copy(self):
        options = defaultdict()
        options.update(self.options)
        options["lightweight"] = True
        learner = Learner(copy.copy(self.layers), options)
        return learner

    def predict(self, X, test=False, until_layer=-1):
        # propagation
        A, _ = self.propagate_and_get(X, test, until_layer)
        
        for i, v in A[-1].items():
            A[-1][i] = v
        return A[-1]

    def propagate_and_get(self, X, test=False, until_layer=-1):
        A = [X]
        E = [X]
        
        #features = X[:] ? why?
        features = X

        for i, layer in enumerate(self.layers[1:]):
            layer_i = i + 1
            if self.options.get("drop_out", False) == True:
                self.__do_dropout(features, i, test)
            features = layer.transform(features)
            E.append(features)
            features = layer.activate(features)
            A.append(features)
            if until_layer==i:
                break

        return A, E

    def __do_dropout(self, features_fmaps, layer_i, test=False):


        drop_out = self.options["drop_out"]
        if type(drop_out) == list:
            drop_out_proba = drop_out[layer_i] if len(drop_out)>layer_i else 0
        elif type(drop_out) == dict:
            drop_out_proba = drop_out[layer_i] if layer_i in drop_out else 0
        else:
            drop_out_proba = drop_out

        for fmap, features in features_fmaps.items():
            if test:
                features_fmaps[fmap] *= drop_out_proba
            else:
                uniform_probas = np.random.random(features.shape)
                drop_out_mask = uniform_probas > drop_out_proba
                features_fmaps[fmap] = features * drop_out_mask
        return features_fmaps

    def get_loss(self, X, y):
        loss_get, _ = loss[self.options["loss_function"]]
        A, _ = self.propagate_and_get(X)
        output = A[-1]
        err = 0.
        for feature_map in  output.keys():
            err += loss_get(y[feature_map], output[feature_map], self.class_weights)
        return err

    def get_deltas(self, X, y, A, E):
        delta = [None] * (len(self.layers))
        
        delta[-1] = {}
        output_layer = self.layers[-1]

        _, loss_get_delta = loss[self.options["loss_function"]]

        for feature_map_out in output_layer.feature_maps:
            delta[-1][feature_map_out] = loss_get_delta(
                    y[feature_map_out], # real
                    A[-1][feature_map_out],  # predicted
                    E[-1][feature_map_out],  # E
                    feature_map_out.activation, 
                    feature_map_out.d_activation,
                    self.class_weights)
            
        for i in xrange(len(delta) - 2, 0, -1):
            delta[i] = {} # we will have a delta of each feature map in layer i
            for feature_map in self.layers[i].feature_maps: # create delta for a feature_map
                nb = 1.
                for to_feature_map in self.layers[i + 1].feature_maps: # find the connections between the current feature map and the feature maps of the next layer
                    if feature_map in to_feature_map.M.keys(): # if feature_map is connected with to_feature_map
                        W = to_feature_map.M[feature_map]
                        D = W.get_delta(delta[i + 1][to_feature_map], A[i][feature_map], E[i + 1][to_feature_map])
                        if feature_map in delta[i]:
                            delta[i][feature_map] += D #/ len(to_feature_map.M.keys())
                            nb += 1
                        else:
                            delta[i][feature_map] = D #/ len(to_feature_map.M.keys())

                delta[i][feature_map] *= linearize(feature_map.d_activation(E[i][feature_map]))
            """
            for to_feature_map in self.layers[i + 1].feature_maps:
                nb = len(to_feature_map.M.keys())
                for feature_map in to_feature_map.M.keys():
                    delta[i][feature_map] /= nb
            """

        return delta

    def get_gradients(self, delta, X, y, A, E):
        deltasW = [None] * len(self.layers)
        deltasB = [None] * len(self.layers)
        for i in xrange(1, len(self.layers)):
            deltasW[i] = {}
            deltasB[i] = {}
            for feature_map in self.layers[i].feature_maps:
                deltasW[i][feature_map] = {}
                deltasB[i][feature_map] = {}
                nb = len(feature_map.M.keys())
                nb = 1.
                for from_feature_map, W in feature_map.M.items():
                    deltaW, deltaB = W.get_gradients(delta[i][feature_map], A[i - 1][from_feature_map])
                    if deltaW is not None:
                        deltasW[i][feature_map][from_feature_map] = deltaW / nb
                    if deltaB is not None:
                        deltasB[i][feature_map][from_feature_map] = deltaB / nb

            # Check gradients
            if self.options["gradient_check"] == True:

                for feature_map in self.layers[i].feature_maps:
                    
                    Ws = defaultdict(list)
                    for from_feature_map in feature_map.M.keys():
                        W = feature_map.M[from_feature_map]
                        Ws[W].append(from_feature_map)
                    for W, feature_maps in Ws.items():
                        if W.W is not None:
                            noise = np.zeros(W.size).reshape((np.prod(W.size),))
                            grads = np.zeros(noise.size)
                            for j in xrange(len(noise)):
                                noise[j] = Learner.EPSILON
                                noise_reshaped = noise.reshape(W.size)

                                initial_W = W.W
                                W.W = initial_W + noise_reshaped
                                new_loss1 = self.get_loss(X, y)
                                W.W = initial_W - noise_reshaped
                                new_loss2 = self.get_loss(X, y)
                                grads[j] = (new_loss1 - new_loss2) / (2*noise[j])
                                W.W = initial_W
                                noise[j] = 0.
                            #grads = grads.reshape(W.size)
                            
                            dw = np.zeros(grads.shape)

                            for fmap in feature_maps:
                                dw += deltasW[i][feature_map][fmap].reshape(grads.shape)
                            #grads2 = grads**2
                            #grads2 += (grads2==0)
                            #max_error = np.sqrt(np.max(((dw - grads) ** 2) / (grads2)))
                            max_error = None
                            for k in xrange(len(grads)):
                                if grads[k]!=0:
                                    max_error = max(max_error, np.abs(dw[k]-grads[k]) / np.abs(grads[k]))
                                else:
                                    max_error = max(max_error, dw[k])
                            if np.all(grads==0):
                                print "Warn : all gradients are zero"

                            print "(%s->%s)Gradient cheking MAX error : %f" % (feature_maps, feature_map,(max_error),)
                            """
                            print deltasB[i][feature_map][from_feature_map]
                            W += (None, Learner.EPSILON)
                            l = self.get_loss(X, y)
                            W -= (None, 2*Learner.EPSILON)
                            l2 = self.get_loss(X, y)
                            print (l - l2) / (2 * Learner.EPSILON)
                            """

        return deltasW, deltasB

    def get_gradients_directly(self, X, y):
        A, E = self.propagate_and_get(X)

        deltas = self.get_deltas(X, y, A, E)
        deltasW, deltasB = self.get_gradients(deltas, X, y, A, E)
        return deltasW, deltasB

    def train(self, X, y, it=0):
        deltasW, deltasB = self.get_gradients_directly(X, y)
        self.update_weights(deltasW, deltasB, it=it)
        return self.get_loss(X, y)
        
    def __get_evolving_learning_rate(self, iteration, lambda0, lambda_=1.):
        #return lambda0
        return lambda0 * (1. / (1 + lambda0 * lambda_ * iteration))

    def update_weights(self, deltasW, deltasB, it=0):
        #alpha = self.__get_evolving_learning_rate(it, self.alpha, (self.lambda2 if self.lambda2 > 0 else 1.))
        alpha = self.alpha
        momentum = self.momentum
        #momentum =  self.__get_evolving_learning_rate(it, self.momentum)

        for i in xrange(1, len(self.layers)):
            deltaW = deltasW[i]
            deltaB = deltasB[i]
            if self.old_gradients[i] is None:
                self.old_gradients[i] = {}
            for feature_map in self.layers[i].feature_maps:
                if feature_map not in self.old_gradients[i]:
                    self.old_gradients[i][feature_map] = {}
                for from_feature_map in feature_map.M.keys():
                    M = feature_map.M[from_feature_map]
                    if feature_map in deltaW and from_feature_map in deltaW[feature_map]:
                        dw = deltaW[feature_map][from_feature_map]
                    else:
                        dw = 0

                    if feature_map in deltaB and from_feature_map in deltaB[feature_map]:
                        db = deltaB[feature_map][from_feature_map]
                    else:
                        db = 0

                    grdW, grdB = 0, 0
                    grdW += dw
                    grdB += db

                    # regu2
                    if M.W is not None: grdW += 2 * self.lambda2 *  M.W
                    if M.b is not None: grdB += 2 * self.lambda2 * M.b
                    # regu1
                    if M.W is not None: grdW += self.lambda1 * (1* (M.W>0) + (-1) * (M.W<0))
                    if M.b is not None: grdB += self.lambda1 * (1* (M.b>0) + (-1) * (M.b<0))
                    
                    """
                    try:
                        old_gradients = self.old_gradients[i][feature_map][from_feature_map]
                    except:
                        old_gradients = None
                    
                    if old_gradients is not None:
                        old_deltaW, old_deltaB = old_gradients
                    else:
                        old_deltaW = 0
                        old_deltaB = 0
                    

                    if dw is not None:
                        meanW = momentum*old_deltaW + grdW * (1. - momentum)
                    else:
                        meanW = 0
                    if db is not None:
                        meanB = momentum*old_deltaB + grdB * (1. - momentum)
                    else:
                        meanB = 0
                    

                    grdW *= alpha
                    grdB *= alpha
                    """

                    try:
                        self.velocity_W[i][feature_map][from_feature_map] = (
                            momentum * self.velocity_W[i][feature_map][from_feature_map] - alpha * grdW)
                    except:
                        if self.velocity_W[i] is None:
                            self.velocity_W[i] = {}
                        if feature_map not in self.velocity_W[i]:
                            self.velocity_W[i][feature_map] = {}
                        if from_feature_map not in self.velocity_W[i][feature_map]:
                            self.velocity_W[i][feature_map][from_feature_map] = -alpha * grdW
                    
                    try:
                        self.velocity_b[i][feature_map][from_feature_map] = (
                            momentum * self.velocity_b[i][feature_map][from_feature_map] - alpha * grdB)
                    except:
                        if self.velocity_b[i] is None:
                            self.velocity_b[i] = {}
                        if feature_map not in self.velocity_b[i]:
                            self.velocity_b[i][feature_map] = {}
                        if from_feature_map not in self.velocity_b[i][feature_map]:
                            self.velocity_b[i][feature_map][from_feature_map] = -alpha * grdB
 
                    velW = self.velocity_W[i][feature_map][from_feature_map]
                    velB = self.velocity_b[i][feature_map][from_feature_map]
                    feature_map.M[from_feature_map] += (velW, velB)
                    

                
    def train_batches(self, X, y, batch_size, it=0):
        
        nb_examples = (X[self.layers[0].feature_maps[0]].shape[0])
        feature_maps_batches_X = {}
        for input_feature_map in self.layers[0].feature_maps:
            feature_maps_batches_X[input_feature_map] = get_batches(X[input_feature_map], batch_size=batch_size)
        feature_maps_batches_y = {}
        for output_feature_map in self.layers[-1].feature_maps:
            feature_maps_batches_y[output_feature_map] = get_batches(y[output_feature_map], batch_size=batch_size)


        nb_batches = len(feature_maps_batches_X[self.layers[0].feature_maps[0]])
        batches_X = []
        batches_Y = []
        
        
        for i in xrange(nb_batches):
            batch_X = {}
            for input_feature_map in self.layers[0].feature_maps:
                batch_X[input_feature_map] = feature_maps_batches_X[input_feature_map][i]
            batch_y = {}
            for output_feature_map in self.layers[-1].feature_maps:
                batch_y[output_feature_map] = feature_maps_batches_y[output_feature_map][i]
            batches_X.append(batch_X)
            batches_Y.append(batch_y)

        if config.options["PARALLELISM"]==True and self.workers is not None:
            nb  = config.options["PARALLEL_POWER"]
            copy_learner = self.get_lightweight_copy()
            results = self.workers.map(zip(repeat(copy_learner, nb), repeat(X, nb), repeat(y, nb), repeat(nb_batches, nb)))
            
            #self.do(lambda me:me.fill(0))

            for learner_ in results:
                self.do_with(learner_, lambda me, other: me + other)

            self.do(lambda me: me / (len(results) +1)) 
            return self.get_loss(X, y)
        else:
            err = 0
            for batch_X, batch_y in zip(batches_X, batches_Y):
                self.train(batch_X, batch_y, it=it)

            return self.get_loss(X, y)

    def from_dataset_to_data(self, dataset):
        input_layer = self.layers[0]
        output_layer = self.layers[-1]
        first_feature_map_input = input_layer.feature_maps[0]
        first_feature_map_output = output_layer.feature_maps[0]
        if len(dataset.input.shape) == len(first_feature_map_input.dimensions) + 1:
            inps = np.expand_dims(dataset.input, axis=1)
        else:
            inps = dataset.input
        

        if dataset.output is not None:
            if len(dataset.output.shape) == len(first_feature_map_output.dimensions) + 1:
                outs = np.expand_dims(dataset.output, axis=1)
            else:
                outs = dataset.output

        inputs = {}
        for i, feature_map in enumerate(input_layer.feature_maps):
            inputs[feature_map] = inps[:, i]

        outputs = {}

        if len(outs.shape)>=1:
            for i, feature_map in enumerate(output_layer.feature_maps):
                outputs[feature_map] = outs[:, i]

        return inputs, outputs 


       



