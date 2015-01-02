from feature_map import FeatureMap, Layer
from connections import ConnectorMaxpooling, ConnectorConvolution, ConnectorWeightedPooling, ConnectorAveragePooling, ConnectorConcatenate
from error_rate import classification_error_rate, regression_error
import numpy as np
from error_rate import confusion_matrix
from utils import Stats
from dataset import Dataset
import copy
import config

def build_convpool_layers(input_layer, conv_size, conv_stride, pool_size, 
                          nbconv=1, layer=1, all_all=True, activation_conv="relu", activation_pool="relu"):
    idim = np.array(input_layer.feature_maps[0].dimensions)
    cdim = np.array(conv_size)

    if pool_size is not None:
        pdim = np.array(pool_size)
    else:
        pdim = None
    sdim = np.array(conv_stride)
    #print "-->"
    #print list(((idim - cdim) / sdim) + 1)
    #if pdim is not None:
    #    print "-->"
    #    print list((((idim - cdim) / sdim) + 1) / pdim)
    
    convs = []  
    concats = []
    pools = []
    
    if len(input_layer.feature_maps) > 1:
        d = tuple([len(input_layer.feature_maps)] + list(input_layer.feature_maps[0].dimensions))
        concat_fmap = FeatureMap(d, name="concat_fmap")
        concats.append(concat_fmap)
        
        conv_dimensions = tuple([1] + list((idim - cdim) / sdim + 1))
        conv_size = tuple([len(input_layer.feature_maps)] + list(conv_size))
        conv_stride = tuple([1] + list(conv_stride))


        if pdim is not None:
            pool_dimensions  = tuple([1] + list(((idim - cdim) / sdim + 1) / pdim))
            pool_size = tuple([1] + list(pool_size))
        else:
            pool_dimensions = None
    else:
        conv_dimensions = (idim - cdim) / sdim + 1
        if pdim is not None:
            pool_dimensions = conv_dimensions / pdim
        else:
            pool_dimensions = None
    
    if pool_dimensions is not None:
        pool_stride = tuple([1] * len(pool_dimensions))
    else:
        pool_stride = None

    for i in xrange(nbconv):
        if len(input_layer.feature_maps) > 1:
            for j in xrange(len(input_layer.feature_maps)):
                ConnectorConcatenate(j, len(input_layer.feature_maps), input_layer.feature_maps[j], concat_fmap).connect()
                conv = FeatureMap(conv_dimensions, activation_func_name=activation_conv, name="conv%d/layer%d" % (i, layer))
            ConnectorConvolution(conv_size, conv_stride, 
                                 concat_fmap, conv).connect()
        else:
            conv = FeatureMap(conv_dimensions, activation_func_name=activation_conv, name="conv%d/layer%d" % (i, layer))
            if all_all:
                for inp in input_layer.feature_maps:
                    ConnectorConvolution(conv_size, conv_stride, inp, conv).connect()
            else:
                ConnectorConvolution(conv_size, conv_stride, input_layer.feature_maps[i], conv).connect()
        
        
        convs.append(conv)
        if pdim is not None:
            pool = FeatureMap(pool_dimensions, name="pool%d/layer%d"%(i, layer), activation_func_name=activation_pool)
            pools.append(pool)
            #ConnectorAveragePooling(2, pool_size, (1, 1, 1), conv, pool).connect()
            ConnectorMaxpooling(pool_size, pool_stride, conv, pool).connect()

    if pdim is None:
        layer_pools = None
    else:
        layer_pools = Layer(pools)
    if len(concats) == 0:
        layer_concats = None
    else:
        layer_concats = Layer(concats)
    return Layer(convs), layer_concats, layer_pools


def build_weighted_pool_layer(input_layer, pool_size, nbpool=1, layer=1, all_all=True):
    py, px = pool_size
    iy, ix = input_layer.feature_maps[0].dimensions
    weight_pool_fmaps = []
    for i in xrange(nbpool):
        weight_pool_fmap = FeatureMap((iy/py, ix/px), 
                                      name="pool%d/Layer%d" % (i+1, layer),
                                      activation_func_name="sigmoid")
        weight_pool_fmaps.append(weight_pool_fmap)

        if all_all:
            for input_fmap in input_layer.feature_maps:
                ConnectorWeightedPooling(pool_size, input_fmap, weight_pool_fmap).connect()
        else:
            input_fmap = input_layer.feature_maps[i]
            ConnectorWeightedPooling(pool_size, input_fmap, weight_pool_fmap).connect()
    return Layer(weight_pool_fmaps)


def batch_training(learner, train_ds, test_ds, nb_epochs, batch_size, update_per_epoch=1, error_rate_min=None, stats=None):
    output_fmap = learner.layers[-1].feature_maps[0]

    data_i, data_o = learner.from_dataset_to_data(train_ds)
    test_data_i, test_data_o = learner.from_dataset_to_data(test_ds)
    
    min_test_error_rate = 1.
    min_train_error_rate = 1.
    min_train_layers = None
    min_test_layers = None

    for i in xrange(nb_epochs):
        print "iteration : %d" % (i,)
        print learner.train_batches(data_i, data_o, batch_size=batch_size, it=i)
        if i % update_per_epoch == 0:
            print
            out = learner.predict(data_i, test=True)
            
            err = classification_error_rate if config.options.get("PROBLEM")=="classification" else regression_error
            train_error_rate  = err(data_o[output_fmap], out[output_fmap])
            print "Overall train error rate : %f" % (train_error_rate,)

            if config.options.get("PROBLEM")=="classification":
                m = confusion_matrix(data_o[output_fmap], out[output_fmap])
                print "train confusion matrix : "
                print m
        
            out = learner.predict(test_data_i, test=True)
            test_error_rate = err(test_data_o[output_fmap], out[output_fmap])

            if train_error_rate < min_train_error_rate:
                min_train_error_rate = train_error_rate
                min_train_layers = copy.deepcopy(learner.layers)
            if test_error_rate < min_test_error_rate:
                min_test_error_rate = test_error_rate
                min_test_layers = copy.deepcopy(learner.layers)

            print "Overall test error rate : %f" % (test_error_rate,)

            if config.options.get("PROBLEM") == "classification":
                m = confusion_matrix(test_data_o[output_fmap], out[output_fmap])
                print "test confusion matrix : "
                print m
                print
            if stats is not None:
                stats.new_point("train_error_rate", (i, train_error_rate))
                stats.new_point("test_error_rate", (i, test_error_rate))

            if error_rate_min is not None and test_error_rate <= error_rate_min:
                break
    return min_train_layers, min_test_layers

from itertools import chain
def cross_validation(learner, ds, nb_batches, nb_elements_min, stats):
    nb_examples = ds.input.shape[0]

    #layers = learner.layers
    
    nb = 0
    step = nb_examples / nb_batches
    min_train_layers = None
    min_test_layers = None
    min_train_error_rate = 1.
    min_test_error_rate = 1.
    layers = []
    for first in xrange(0, nb_examples, step):
        last = min(first + step, nb_examples)
        last_one = False
        if (min(first+step+step, nb_examples) - min(first + step, nb_examples) + 1) < nb_elements_min:
            last = nb_examples
            last_one = True

        print "BATCH %d-%d" % (first, last)
        test_ds_ = Dataset(ds.input[first:last], ds.output[first:last])
        i = [ds.input[idx] for idx in chain(xrange(0, first), xrange(last, ds.input.shape[0]))]
        o = [ds.output[idx] for idx in  chain(xrange(0, first), xrange(last, ds.input.shape[0]))]
        train_ds_ = Dataset(i, o)
        
        #learner.layers = copy.deepcopy(layers)
        learner.init()

        local_stats = Stats()
        min_train_layers_local, min_test_layers_local = batch_training(learner, train_ds_, test_ds_, stats=local_stats, **config.options["BATCH_TRAINING"])

        best_test_error_rate = min(map(lambda p:p[1],  local_stats.get_points("test_error_rate")) )
        if best_test_error_rate < min_test_error_rate:
            min_test_error_rate = best_test_error_rate
            min_test_layers = min_test_layers_local

        best_train_error_rate = min(map(lambda p:p[1], local_stats.get_points("train_error_rate") ))
        if best_train_error_rate < min_train_error_rate:
            min_train_error_rate = best_train_error_rate
            min_train_layers = min_train_layers_local
        layers.append(min_test_layers_local)
        stats.new_point("cross_validation_train_error_rate", (nb, best_train_error_rate))
        stats.new_point("cross_validation_test_error_rate", (nb, best_test_error_rate))
        if last_one:
            break
        nb += 1
    return min_train_layers, min_test_layers, layers
