import numpy as np
from utils import linearize, Workers, safe_log, safe_exp
from multiprocessing import Pool, cpu_count
import config
import copy
import signal
from itertools import repeat
import sys
#from scipy.signal import convolve2d, fftconvolve
from utils import pad, pad_to_desired_shape
class Connector(object):

    def __init__(self, from_fmap, to_fmap):
        self.from_fmap = from_fmap
        self.to_fmap = to_fmap

    def connect(self):
        raise NotImplementedError

class ConnectorAll(Connector):

    def __init__(self, *args, **kwargs):
        super(ConnectorAll, self).__init__(*args, **kwargs)

    def connect(self):
        #1+... , why 1?bias
        size = (self.to_fmap.get_nb_features(), 1+self.from_fmap.get_nb_features())
        W = UsualMatrix(size=size, output_size=self.to_fmap.dimensions)
        self.to_fmap.M[self.from_fmap] = W

class ConnectorKernel(Connector):

    def __init__(self, kernel_size, stride, *args, **kwargs):

        super(ConnectorKernel, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride


class AbstractWeightMatrix(object):
    def __init__(self):
        self.W = None
        self.b = None
    
    def init_W_from_size(self, size):
        self.W = np.random.normal(0., 1. / np.sqrt(np.prod(size)), size=size)
        #self.W = np.random.uniform(-np.sqrt(np.prod(size)), np.sqrt(np.prod(size)), size=size)
    
    def init(self):
        pass

    def transform(self, X):
        raise NotImplementedError

    def fill(self, value):
        if hasattr(self.W, "fill"): 
            self.W.fill(value)
        else:
            self.W = value
        if hasattr(self.b, "fill"):
            self.b.fill(value)
        else:
            self.b = value
        return self

    def __isub__(self, gradients):
        if isinstance(gradients, AbstractWeightMatrix):
            deltaW, deltaB = gradients.W, gradients.b
        else:
            deltaW, deltaB = gradients
        if self.W is not None and deltaW is not None:
            self.W -= deltaW
        if self.b is not None and deltaB is not None: 
            self.b -= deltaB
        return self

    def __iadd__(self, gradients):

        if isinstance(gradients, AbstractWeightMatrix):
            deltaW, deltaB = gradients.W, gradients.b
        else:
            deltaW, deltaB = gradients
        if self.W is not None and deltaW is not None : self.W += deltaW
        if self.b is not None and deltaB is not None : self.b += deltaB
        return self
    
    def __add__(self, gradients):
        return self.__iadd__(gradients)

    def __idiv__(self, scalar):
        if self.W is not None:
            self.W /= scalar
        if self.b is not None:
            self.b /= scalar
        return self
    
    def __div__(self, scalar):
        return self.__idiv__(scalar)

    def __imul__(self, scalar):
        if self.W is not None:
            self.W *= scalar
        if self.b is not None:
            self.b *= scalar
        return self

    def __mul__(self, scalar):
        return self.__imul__(scalar)


class UsualMatrix(AbstractWeightMatrix):

    def __init__(self, size=None, output_size=None, from_=None):
        super(UsualMatrix, self).__init__()
        self.size = size
        self.output_size = output_size
        if from_ is None:
            self.init()
        else:
            self.W = copy.copy(from_)

    def init(self):
        self.init_W_from_size(self.size)
        #self.W[:, -1] = 0.


    def transform(self, X):
        X = linearize(X)
        X = np.concatenate((X.T, np.ones((1, X.shape[0]))), axis=0).T
        return ((np.dot(self.W, X.T) ).T).reshape([X.shape[0]] + list(self.output_size))

    def get_delta(self, delta_next_layer, A, E_next_layer):
        delta =  np.dot(delta_next_layer, self.W)
        delta = delta[:, 0:-1]
        return delta

    def get_gradients(self, delta, A_previous_layer):
        A = A_previous_layer.reshape((A_previous_layer.shape[0], np.prod(A_previous_layer.shape[1:])))
        A = np.concatenate((A.T, np.ones((1, A.shape[0]))), axis=0).T
        deltaW = np.dot(delta.T, A)
        return deltaW, None

    def __getslice__(self, a, b):
        return self.W.__getslice__(a, b)


class ConvolutionalWeightMatrix(AbstractWeightMatrix):

    def __init__(self, size, stride=(1, 1), input_size=None):
        super(ConvolutionalWeightMatrix, self).__init__()
        self.size = size
        self.input_size = input_size
        self.stride = stride
        self.output = None
        self.delta = None
        self.deltaW = None
        self.init()
    
    def init(self):
        self.init_W_from_size(self.size)
        self.b = 0

    def transform(self, X):
        assert len(X.shape)-1 == len(self.size)
        assert (np.array(X.shape[1:]) >= np.array(self.size)).all()
        output_dim = self.__get_output_size(X.shape)

        if self.output is None or output_dim != self.output.shape:
            self.output = np.zeros(output_dim)
        self.output.fill(0)
        #for i in xrange(X.shape[0]):
        #   output[i] = fftconvolve(X[i], self.W, 'valid')
        
        #output = fftconvolve(X, self.W, 'valid')
        self.__fill_output(X, self.output, [], output_dim[1:])
        #print np.sum(output[0])
        return self.output

    def __get_output_size(self, input_size):
        output_dim = ([input_size[0]]  + list((( np.array(input_size[1:]) - 
                                        np.array(self.size)) / np.array(self.stride)) + 1))
        return output_dim
    
    def __fill_output(self, X, output, coords, dimensions):

        if len(self.size) == 2:
            for y in xrange(output.shape[1]):
                for x in xrange(output.shape[2]):
                    X_ = (X[:, y*self.stride[0]:y*self.stride[0] + self.size[0], x*self.stride[1]:x*self.stride[1] + self.size[1]] * self.W)
                    #X_ =  np.apply_over_axes(np.sum, X_, axes=[1, 2])
                    X_ = np.sum(X_, axis=(1, 2))
                    output[:, y, x] = X_
            output += self.b
            return output
        
        if len(dimensions) == 0:
            nb_examples = X.shape[0]
            coords_ = [0] + coords
            coords = list(np.array(coords) * np.array(self.stride))
            coords_next = [nb_examples] + (
                list(np.array(coords) + np.array(list(np.array(self.size)))))
            coords = [0] + (coords)
            nsquare = X[map(lambda d:slice(*d), zip(coords, coords_next))]
            nsquare = pad_to_desired_shape(nsquare, [(cn - c) for c, cn in zip(coords, coords_next)], 'constant')

            output_coords_next = list(np.array(coords_) + 1)
            #output_coords_next[0] = nb_examples
            #output_coords = map(lambda d:slice(*d), zip(coords_, output_coords_next))

            output_coords = coords_
            output_coords[0] = slice(0, nb_examples)
            
            #output[output_coords] = np.apply_over_axes(np.sum, 
            #                                          nsquare*self.W, axes=range(1, len(nsquare.shape)) ) + self.b
            output[output_coords] = np.sum(nsquare * self.W, axis=tuple(range(1, len(nsquare.shape))) ) + self.b
            return
        for d in xrange(dimensions[0]):
            self.__fill_output(X, output, coords + [d], dimensions[1:])

    def get_delta(self, delta_next_layer, A, E_next_layer):
        nb_examples = delta_next_layer.shape[0]
        input_size = [nb_examples] + list(self.input_size)
        output_dim = self.__get_output_size(input_size)

        delta_next_layer = delta_next_layer.reshape(output_dim)
        
        if self.delta is None or input_size != self.delta.shape:
            self.delta = np.zeros(input_size)
        self.delta.fill(0)
        self.__fill_delta(self.delta, delta_next_layer, [], delta_next_layer.shape[1:])
        return linearize(self.delta)

    def __fill_delta(self, delta, delta_next_layer, coords, dimensions):
        # 2D
        if len(self.size) == 2:
            for y in xrange(delta_next_layer.shape[1]):
                for x in xrange(delta_next_layer.shape[2]):
                    delta[:, y*self.stride[0]:y*self.stride[0] + self.size[0], x*self.stride[1]:x*self.stride[1] + self.size[1]] += self.W * delta_next_layer[:, y:y+1, x:x+1]
            return
        if len(dimensions) == 0:
            nb_examples = (delta.shape[0])
            

            next_layer_coords_slices = [0] + coords
            next_layer_coords_slices[0] = slice(0, nb_examples)
            next_layer_coords_slices[1:] = map(lambda d:slice(d, d+1), next_layer_coords_slices[1:])

            coords = [0] + list(coords*np.array(self.stride))
            coords_next = list(np.array(coords) + np.array([nb_examples] + list(np.array(self.size))))

            coords_slices = map(lambda d:slice(*d), zip(coords,  coords_next))

            delta_next_layer_ = delta_next_layer[next_layer_coords_slices]

            delta[coords_slices] += self.W * delta_next_layer_
    
            return
        for d in xrange(dimensions[0]):
            self.__fill_delta(delta, delta_next_layer, coords + [d], dimensions[1:])

    def get_gradients(self, delta, A_previous_layer):
        input_size = [delta.shape[0]] + list(self.input_size)
        output_size = self.__get_output_size(input_size)
        delta = delta.reshape([delta.shape[0]] + output_size[1:])
        
        if self.deltaW is None:
            self.deltaW = np.zeros(self.size)
        self.deltaW.fill(0)
        self.__fill_deltaW(self.deltaW, delta, A_previous_layer, [], delta.shape[1:])
        deltaB = np.sum(delta)
        return [self.deltaW, deltaB]

    def __fill_deltaW(self, deltaW, delta, A_previous_layer, coords, dimensions):
        if len(self.size) == 2:
            for y in xrange(delta.shape[1]):
                for x in xrange(delta.shape[2]):
                    deltaW += np.sum(
                        A_previous_layer[:, y*self.stride[0]:y*self.stride[0] + self.size[0], x*self.stride[1]:x*self.stride[1] + self.size[1]] *
                        delta[:, y:y+1, x:x+1], axis=0) 
            #deltaW = 0
            return
        if len(dimensions) == 0:
            nb_examples = (delta.shape[0])

            next_layer_coords_slices = [0] + coords
            next_layer_coords_slices[0] = slice(0, nb_examples)
            next_layer_coords_slices[1:] = map(lambda d:slice(d, d+1), next_layer_coords_slices[1:])

            coords_input = [0] + list(np.array(coords) * np.array(self.stride))
            coords_input_next = [nb_examples] + list((coords_input[1:]) + np.array((self.size)))

            coord_slices = map(lambda d:slice(*d), zip(coords_input, coords_input_next))
            D = np.sum(A_previous_layer[coord_slices] * delta[next_layer_coords_slices], axis=0)
            #deltaW[  [slice(0, d) for d in D.shape] ] += D
            #deltaW += D
            deltaW += pad_to_desired_shape(D, deltaW.shape, 'constant')
            return
        for d in xrange(dimensions[0]):
            self.__fill_deltaW(deltaW, delta, A_previous_layer, coords + [d], dimensions[1:])


class WeightedPoolingMatrix(AbstractWeightMatrix):

    def __init__(self, size):
        self.size = size
        self.init_W_from_size(self.size)
        self.b = 0

    def transform(self, X):
        output = np.zeros(self.__get_output_size(X))

        if len(self.size)==2:
            sy, sx = self.size
            for y in xrange(output.shape[1]):
                for x in xrange(output.shape[2]):
                    output[:, y:y+1, x:x+1] = np.apply_over_axes(np.sum,
                                                         X[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx] * self.W,
                                                         axes=[1, 2])
        elif len(self.size)==3:
            sz, sy, sx = self.size
            for z in xrange(output.shape[1]):
                for y in xrange(output.shape[2]):
                    for x in xrange(output.shape[3]):
                        output[:, z:z+1, y:y+1, x:x+1] = np.apply_over_axes(np.sum,
                                                             X[:, z*sz:(z+1)*sz, y*sy:(y+1)*sy, x*sx:(x+1)*sx] * self.W,
                                                             axes=[1, 2, 3])

        output += self.b
        return output

    def get_delta(self, delta_next_layer, A, E_next_layer):
        output_size = self.__get_output_size(A)
        delta_next_layer = delta_next_layer.reshape(output_size)

        delta = np.zeros(A.shape)

        for i in xrange(delta.shape[0]):
            delta[i] = np.kron(delta_next_layer[i], self.W)

        return delta.reshape((output_size[0], np.prod(delta.shape[1:])))

    def get_gradients(self, delta, A_previous_layer):
        output_size = self.__get_output_size(A_previous_layer)

        delta = delta.reshape(output_size)
        deltaW = np.zeros(self.size)

        if len(self.size)==2:
            sy, sx = self.size
            for y in xrange(delta.shape[1]):
                for x in xrange(delta.shape[2]):
                    A_ = A_previous_layer[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx]
                    deltaW += np.sum(delta[:, y:y+1, x:x+1] * A_, axis=0)
                    #deltaW += np.sum(np.kron(A_, delta[:, y:y+1, x:x+1]), axis=0)
        elif len(self.size)==3:
            sz, sy, sx = self.size
            for z in xrange(delta.shape[1]):
                for y in xrange(delta.shape[2]):
                    for x in xrange(delta.shape[3]):
                        A_ = A_previous_layer[:, z*sz:(z+1)*sz, y*sy:(y+1)*sy, x*sx:(x+1)*sx]
                        deltaW += np.sum(delta[:, z:z+1, y:y+1, x:x+1] * A_, axis=0)

        deltaB = np.sum(delta)
        return deltaW, deltaB

    def __get_output_size(self, X):
        return [X.shape[0]] + list(np.array(X.shape[1:]) / np.array(self.size))



class PoolingMatrix(AbstractWeightMatrix):
    def __init__(self, size, alpha=15):
        self.size = size
        self.input_size = None
        self.alpha = alpha
        self.W = None
        self.b = None

    def transform(self, X):
        output_size = self.__get_output_size(X.shape)
        output = np.zeros(output_size)
        self.__fill_output(X, output, [], output_size[1:])
        return output

    def __get_output_size(self, input_size):
        return [input_size[0]] + list(np.array(input_size[1:]) / np.array(self.size))

    def __fill_output(self, X, output, coords, dimensions):

        if len(self.size) == 2:
            sy, sx = self.size
            for y in xrange(output.shape[1]):
                for x in xrange(output.shape[2]):
                    X_ = X[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx]
                    
                    #X_max = np.apply_over_axes(np.max, X_, axes=[1, 2])
                    X_max = np.max(X_, axis=(1, 2))
                    output[:, y, x] = X_max
            return
        if len(dimensions) == 0:
            nb_examples = X.shape[0]

            coords_input = [0] + list(np.array(coords) * np.array(self.size))
            coords_input_next = [nb_examples] + list((np.array(coords)+1) * np.array(self.size))
            coords_input_slices = [slice(*d) for d in zip(coords_input, coords_input_next)]

            coords_output = [slice(0, nb_examples)] + coords

            X = X[coords_input_slices]
            X = X.reshape((nb_examples, np.prod(X.shape[1:])))
            
            output[coords_output] = np.max(X, axis=1)
        
            return
        for d in xrange(dimensions[0]):
            self.__fill_output(X, output, coords + [d], dimensions[1:])

    def get_delta(self, delta_next_layer, A, E_next_layer):
        delta_size = [delta_next_layer.shape[0]] + list(self.input_size)
        delta = np.zeros(delta_size)
        
        output_size = self.__get_output_size(delta_size)
        nb_examples = delta_next_layer.shape[0]
        delta_next_layer = delta_next_layer.reshape(output_size)
        
        delta = np.zeros(delta_size)
        self.__fill_delta(delta, delta_next_layer, A, [], output_size[1:])
        return linearize(delta)

    def __fill_delta(self, delta, delta_next_layer, A, coords, dimensions):

        if len(self.size) == 2:
            sy, sx = self.size
            for y in xrange(delta_next_layer.shape[1]):
                for x in xrange(delta_next_layer.shape[2]):
                    A_ = A[:, (y*sy):((y+1)*sy), (x*sx):((x+1)*sx)]
                    
                    A__ = A_.reshape((delta_next_layer.shape[0], sx*sy))
                    maxes = A__.argmax(axis=1)[:, np.newaxis]
                    _, f = np.indices(A__.shape)
                    D = delta_next_layer[:, y:y+1, x:x+1]

                    R =  D * ((f == maxes).reshape( (delta_next_layer.shape[0], sy, sx) ))
                    delta[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx] += R
            return

        if len(dimensions) == 0:
            nb_examples = delta.shape[0]
            coords_input = [0] + list(np.array(coords) * np.array(self.size))
            coords_input_next = [nb_examples] + list((np.array(coords)+1) * np.array(self.size))
            coords_input_slices = [slice(*d) for d in zip(coords_input, coords_input_next)]
            coords_output = [slice(0, nb_examples)] + coords
        
            A_ = A[coords_input_slices]
            A__ = linearize(A_)
            maxes = A__.argmax(axis=1)[:, np.newaxis]
            _, f = np.indices(A__.shape)

            D = (delta_next_layer[tuple(coords_output)]).reshape([nb_examples] + [1] * len(coords))
            R = D * ((f == maxes).reshape(A_.shape))
            delta[coords_input_slices] += R
            return
        for d in xrange(dimensions[0]):
            self.__fill_delta(delta, delta_next_layer, A, coords + [d], dimensions[1:])

    def get_gradients(self, delta, A_previous_layer):
        return [None, None]

    def __isub__(self, delta):
        return self

    def __iadd__(self, delta):
        return self


class AveragePoolingMatrix(AbstractWeightMatrix):
    def __init__(self, size, stride=(1, 1), P=2):
        self.size = size
        self.P = P
        self.stride = stride
        self.input_size = None
        self.W = None
        self.b = None

    def transform(self, X):
        output_size = self.__get_output_size(X.shape)
        output = np.zeros(output_size)
        self.__fill_output(X, output, [], output_size[1:])
        return output

    def __get_output_size(self, input_size):
        return [input_size[0]] + list(np.array(input_size[1:]) / np.array(self.size))

    def __fill_output(self, X, output, coords, dimensions):

        if len(self.size) == 2:
            sy, sx = self.size
            for y in xrange(output.shape[1]):
                for x in xrange(output.shape[2]):
                    X_ = X[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx]
                    #output[:, y:y+1, x:x+1] = np.apply_over_axes(np.sum, X_ ** self.P, axes=[1, 2]) ** (1./self.P)
                    output[:, y, x] = np.sum(X_ ** self.P, axis=(1, 2)) ** (1./self.P)

            return
        if len(dimensions) == 0:
            nbdim = len(X.shape)
            nb_examples = X.shape[0]

            coords_input = [0] + list(np.array(coords) * np.array(self.size))
            coords_input_next = [nb_examples] + list((np.array(coords)+1) * np.array(self.size))
            coords_input_slices = [slice(*d) for d in zip(coords_input, coords_input_next)]

            coords_output = [slice(0, nb_examples)] + [c for c in coords]

            X_ = X[coords_input_slices]
            #output[coords_output] = np.apply_over_axes(np.sum, X_**self.P, axes=range(1, nbdim)) ** (1./self.P)
            output[coords_output] = np.sum(X_**self.P, axis=tuple(range(1, nbdim))) ** (1./self.P)
            return
        for d in xrange(dimensions[0]):
            self.__fill_output(X, output, coords + [d], dimensions[1:])

    def get_delta(self, delta_next_layer, A, E_next_layer):
        delta_size = [delta_next_layer.shape[0]] + list(self.input_size)
        delta = np.zeros(delta_size)
        
        output_size = self.__get_output_size(delta_size)
        nb_examples = delta_next_layer.shape[0]
        delta_next_layer = delta_next_layer.reshape(output_size)
        
        delta = np.zeros(delta_size)
        self.__fill_delta(delta, delta_next_layer, A, E_next_layer, [], output_size[1:])
        return linearize(delta)

    def __fill_delta(self, delta, delta_next_layer, A, E_next_layer, coords, dimensions):

        if len(self.size) == 2:
            sy, sx = self.size
            for y in xrange(delta_next_layer.shape[1]):
                for x in xrange(delta_next_layer.shape[2]):
                    A_ = A[:, (y*sy):((y+1)*sy), (x*sx):((x+1)*sx)]
                    E_ = E_next_layer[:, y:y+1, x:x+1]
                    D = delta_next_layer[:, y:y+1, x:x+1]

                    E__ = E_ + (E_==0)
                    delta[:, y*sy:(y+1)*sy, x*sx:(x+1)*sx] = D * E__**(1.-self.P) * A_**(self.P - 1)
            return

        if len(dimensions) == 0:
            nb_examples = delta.shape[0]
            coords_input = [0] + list(np.array(coords) * np.array(self.size))
            coords_input_next = [nb_examples] + list((np.array(coords)+1) * np.array(self.size))
            coords_input_slices = [slice(*d) for d in zip(coords_input, coords_input_next)]
            coords_output = [slice(0, nb_examples)] + coords
        
            D = (delta_next_layer[tuple(coords_output)]).reshape([nb_examples] + [1] * len(coords))
            A_ = A[coords_input_slices]
            E_ = (E_next_layer[tuple(coords_output)]).reshape([nb_examples] + [1] * len(coords))
            E_ = E_ + (E_==0)
            delta[coords_input_slices] = D * E_**(1-self.P) * A_**(self.P - 1)
            return
        for d in xrange(dimensions[0]):
            self.__fill_delta(delta, delta_next_layer, A, E_next_layer, coords + [d], dimensions[1:])

    def get_gradients(self, delta, A_previous_layer):
        return [None, None]

class ConnectorConvolution(ConnectorKernel):

    def __init__(self, *args, **kwargs):
        super(ConnectorConvolution, self).__init__(*args, **kwargs)

        if self.to_fmap.W is  None:
            self.to_fmap.W = ConvolutionalWeightMatrix(self.kernel_size, self.stride)
            self.to_fmap.W.input_size = self.from_fmap.dimensions


    def connect(self):
        self.to_fmap.M[self.from_fmap] = self.to_fmap.W


class ConnectorMaxpooling(ConnectorKernel):

    def __init__(self, *args, **kwargs):
        super(ConnectorMaxpooling, self).__init__(*args, **kwargs)
        assert np.array_equal(np.array(self.to_fmap.dimensions) * np.array(self.kernel_size), 
                              np.array(self.from_fmap.dimensions))

    
    def connect(self):
        W = PoolingMatrix(self.kernel_size)
        W.input_size = self.from_fmap.dimensions
        self.to_fmap.M[self.from_fmap] = W

        self.to_fmap.activation_func_name = "id"

class ConnectorWeightedPooling(ConnectorKernel):
    
    def __init__(self, *args, **kwargs):
        super(ConnectorWeightedPooling, self).__init__(*args, **kwargs)
        assert np.array_equal(np.array(self.to_fmap.dimensions) * np.array(self.kernel_size), 
                              np.array(self.from_fmap.dimensions))

    def connect(self):
        W = WeightedPoolingMatrix(self.kernel_size)
        self.to_fmap.M[self.from_fmap] = W

class ConnectorAveragePooling(ConnectorKernel):
    
    def __init__(self, P, *args, **kwargs):
        super(ConnectorAveragePooling, self).__init__(*args, **kwargs)
        assert np.array_equal(np.array(self.to_fmap.dimensions) * np.array(self.kernel_size), 
                              np.array(self.from_fmap.dimensions))
        self.P = P

    def connect(self):
        W = AveragePoolingMatrix(self.kernel_size, self.stride, self.P)
        W.input_size = self.from_fmap.dimensions
        self.to_fmap.M[self.from_fmap] = W
        #self.to_fmap.activation_func_name = "sig"


class ConcatenateMatrix(AbstractWeightMatrix):

    def __init__(self, dim_id, nb_dims, output_dim):
        self.dim_id = dim_id
        self.output_dim = output_dim
        self.nb_dims = nb_dims
        self.W = None
        self.b = None

    def transform(self, X):
        return X

    def get_delta(self, delta_next_layer, A, E_next_layer):
        delta_next_layer = delta_next_layer.reshape( (delta_next_layer.shape[0], self.nb_dims, delta_next_layer.shape[1] / self.nb_dims) )
        delta_next_layer = delta_next_layer[:, self.dim_id]
        return linearize(delta_next_layer)

    def get_gradients(self, delta, A_previous_layer):
        return [None, None]

    

class ConnectorConcatenate(Connector):
    
    def __init__(self, dim_id, nb_dims, *args, **kwargs):
        super(ConnectorConcatenate, self).__init__(*args, **kwargs)
        self.dim_id = dim_id
        self.nb_dims = nb_dims
        self.to_fmap.aggreg = "concat" 
        self.to_fmap.activation_func_name = "id"

    def connect(self):
        self.to_fmap.M[self.from_fmap] = ConcatenateMatrix(self.dim_id, self.nb_dims, self.to_fmap.dimensions)

class ConnectorRectification(Connector):
    def __init__(self, *args, **kwargs):
        super(ConnectorConcatenate, self).__init__(*args, **kwargs)

