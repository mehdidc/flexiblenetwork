from utils import import_object, ressource_full_filename
from constants import *
from dataset import Dataset
from feature_map import FeatureMap, Layer
from constants import *
from easy import build_weighted_pool_layer, build_convpool_layers
from itertools import repeat
from connections import ConnectorConvolution, ConnectorAll, ConvolutionalWeightMatrix, ConnectorMaxpooling, ConnectorAveragePooling
import numpy as np
import config
import os
import copy
from connections import UsualMatrix
data_name = config.options.get("DATA").get("NAME", "")

def mlp_from_file(filename, output_size=None, activations=None, input_size=None, **kwargs):
    Ws = np.load(ressource_full_filename(os.path.join(data_name, filename), DATA))
    layers = []
    
    if input_size is None:
        layers.append( Layer( [FeatureMap( (Ws[0].shape[1],) ) ] ) )
    else:
        layers.append(build_input_layer(input_size))

    for W in Ws:
        if len(W.shape) == 1:
            W = W[0]
        layers.append(  Layer( [FeatureMap( (W.shape[0],) )] )  )
        prev, cur = layers[-2].feature_maps[0], layers[-1].feature_maps[0]
        ConnectorAll(prev, cur).connect()
        
        W_ = copy.copy(W)
        W_.resize( (W.shape[0], W.shape[1] + 1) )
        cur.M[prev] = UsualMatrix(size=W_.shape, output_size=(W_.shape[0],), from_=W_)

    if output_size is not None:
        layers.append(  Layer( [FeatureMap(output_size)] ) )
        ConnectorAll(layers[-2].feature_maps[0], layers[-1].feature_maps[0]).connect()

    if activations is not None:
        if type(activations) == list:
            store_activations(layers, activations)
        else:
            store_activations(layers, repeat(activations))

    return layers

def mlp(nbunits, input_dim=3, activations=None, build_input_layer="build_layers.build_input_layer", build_output_layer="build_layers.build_output_layer", output_dim=1, **kwargs):
    if activations is None:
        activations = repeat(None)
    
    input_layer = import_object(build_input_layer)(input_dim)
    
    layers = [input_layer]

    last_layer = input_layer
    for nb in nbunits:
        fc_fmap = FeatureMap((nb,))
        for fmap in last_layer.feature_maps:
            ConnectorAll(fmap, fc_fmap).connect()
        fc_layer = Layer([fc_fmap])
        layers.append(fc_layer)
        last_layer = fc_layer
    
    layer = import_object(build_output_layer)(output_dim)
    for fmap in last_layer.feature_maps:
        ConnectorAll(fmap, layer.feature_maps[0]).connect()
    layers.append(layer)
    
    store_activations(layers, activations)
    return layers


def pseudoconv3D(conv, nbprof=15, fc_layer_nb=10, activations=None, l=30, w=18, h=18, activation="relu", **kwargs):

    if activations is None:
        activations = repeat(None)
    input_layer = build_input_layer(input_dim=2)
    input_layer_fmaps = input_layer.feature_maps
    layers = [input_layer]

    nbconv = l - nbprof + 1
    conv_fmaps = [] 
    cx, cy = conv
    for i in xrange(nbconv):
        conv_fmap = FeatureMap( (h - cy + 1, w - cx + 1), activation_func_name=activation)
        for z in xrange(i, i + nbprof):
            ConnectorConvolution( (cy, cx), (1, 1), input_layer_fmaps[z], conv_fmap).connect()
        conv_fmaps.append(conv_fmap)
    
    conv_layer = Layer(conv_fmaps)
    layers.append(conv_layer)

    output_layer = build_output_layer()
    output_fmap = output_layer.feature_maps[0]
    if fc_layer_nb is not None:
        fc_fmap = FeatureMap((fc_layer_nb,))
        for i in xrange(nbconv):
            ConnectorAll(conv_layer.feature_maps[i], fc_fmap).connect()
        ConnectorAll(fc_fmap, output_fmap).connect()
        layers.append(Layer([fc_fmap]))
        layers.append(output_layer)
    else:
        for i in xrange(nbconv):
            ConnectorAll(conv_layer.feature_maps[i], output_fmap).connect()
        layers.append(output_layer)


    store_activations(layers, activations)
    return layers
    
def conv(convs, strides, pools, nbmaps, fc_layer_nb=None, input_dim=2, activations=None, build_input_layer="build_layers.build_input_layer", build_output_layer="build_layers.build_output_layer", output_dim=1, **kwargs):
    
    if activations is None:
        activations = repeat(None) 
    if pools is None:
        pools = repeat(None)
    input_layer = import_object(build_input_layer)(input_dim)
    layers = [input_layer]
    
    last_layer = input_layer
    i = 0
    for conv, stride, pool, nbmap in zip(convs, strides, pools, nbmaps):
        i += 1
        conv, concats, pool = build_convpool_layers(last_layer, conv, stride, pool, nbconv=nbmap, layer=i, all_all=True)

        if concats is not None:
            layers.append(concats)
            last_layer = concats
        
        layers.append(conv)
        last_layer = conv
        
        if pool is not None:
            layers.append(pool)
            last_layer = pool
    
    output_layer = import_object(build_output_layer)(output_dim)

    if fc_layer_nb is not None:
        if type(fc_layer_nb) != list:
            fc_layer_nb = [fc_layer_nb]

        for nb in fc_layer_nb:
            fc_layer_fmap = FeatureMap((nb,), activation_func_name="relu", name="fc%d"% (i,))
            for fmap in last_layer.feature_maps:
                ConnectorAll(fmap, fc_layer_fmap).connect()

            layer = Layer([fc_layer_fmap])
            layers.append(layer)
            last_layer = layer
        
        
        for fmap in last_layer.feature_maps:
            ConnectorAll(fc_layer_fmap, output_layer.feature_maps[0]).connect()
    else:
        for fmap in last_layer.feature_maps:
            ConnectorAll(fmap, output_layer.feature_maps[0]).connect()
    layers.append(output_layer)


    store_activations(layers, activations)
    return layers


def build_input_layer(input_dim):
    return Layer([FeatureMap( input_dim )])


def build_output_layer(output_dim):
    return Layer(  [FeatureMap (  output_dim , activation_func_name="tanh")]  )

def store_activations(layers, activations):
    for layer, activation in zip(layers[1:-1], activations):
        if activation is None: activation="relu"
        for fmap in layer.feature_maps:
            fmap.activation_func_name = activation
