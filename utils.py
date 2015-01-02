from numpy import prod


import gzip
import copy
import numpy as np
import time
import traceback
import config
import os
from dataset import Dataset
from constants import *

from collections import defaultdict

class Stats(object):
    
    def __init__(self):
        self.data = defaultdict(list)
    
    def subjects(self):
        return self.data.keys()

    def new_point(self, subject, p):
        self.data[subject].append(p)

    def get_points(self, subject):
        return self.data[subject]

def with_job_suffix(s):
    suff = os.getenv("jobid", "")
    return s + suff

def ressource_full_filename(filename, ressource_type, output=False):
    folders = config.options["FOLDERS"][ressource_type]
    for folder in folders:
        full_filename = os.path.join(folder, filename)
        if output == True:
            return full_filename
        if os.path.exists(full_filename):
            return full_filename
    raise Exception("Cannot find %s in %s" % (filename, folders))
    

def linearize(array):
	return array.reshape((array.shape[0], np.prod(array.shape[1:])))

from multiprocessing import Queue, Process, cpu_count, Event
CPUCOUNT_DEFAULT = min(8, cpu_count())
class Workers(object):

	def __init__(self, target, size=None):
		if size is None: size = CPUCOUNT_DEFAULT
		self.size = size
		self.target = target
		self.communication_queue = Queue()
		self.work_queue = Queue()
		self.exit = Event()
		
		def worker():
			while not self.exit.is_set():
				try:
					i, args = self.work_queue.get(block=False)
				except:
					continue				
				value = target(*args)
				self.communication_queue.put((i, value))

		self.procs = []
		for i in xrange(self.size):
			self.procs.append(Process(target=worker))

		for proc in self.procs:
			proc.start()

	def updater(self, all_args, update_function):
		self.__put_on_work_queue(all_args)
		
		for i in xrange(len(all_args)):
			arg_id, value = self.communication_queue.get()
			updates = update_function(arg_id, value)

	def map(self, all_args):

		values = [None] * len(all_args)
		
		self.__put_on_work_queue(all_args)
		for i in xrange(len(all_args)):
			pid, value = self.communication_queue.get()
			values[pid] = value
		return values

	def __put_on_work_queue(self, all_args):
		for i, args in enumerate(all_args):
			if type(args) != list and type(args) != tuple:
				args = [args]
			args = list(args)
			self.work_queue.put((i, args))

	def close(self):
		self.exit.set()
		for p in self.procs:
			p.join()


def get_batches(X, batch_size=None, nb_batches=None):
	batches = []
	if batch_size is not None:
		nb_batches = X.shape[0] / batch_size + (1 if X.shape[0] % batch_size != 0 else 0)
	elif nb_batches is not None:
		batch_size = X.shape[0] / nb_batches
	err = 0.
	for i in xrange(nb_batches):
		b_from, b_to = i * batch_size, min((i + 1) * batch_size, X.shape[0])
		batch_X = X[( slice(b_from,b_to),)]
		batches.append(batch_X)
	return batches

import numpy


def NormalizeInplace(array, imin=0, imax=1):
    dmin = array.min()
    dmax = array.max()
    array -= dmin
    array *= imax - imin
    array /= dmax - dmin
    array += imin

def NormalizeCopy(array, imin=0, imax=1):
    dmin = array.min()
    dmax = array.max()
    return imin + (imax - imin) * (array - dmin) / (dmax - dmin)

def Normalize2dArray(array):
    assert isinstance(array, numpy.ndarray) and len(array.shape) == 2
    res = []
    for row in array.astype(float):
        tmp = (row.astype(float) - row.astype(float).min()) / (row.astype(float).max() - row.astype(float).min())
        assert tmp.min() == 0. and tmp.max() == 1.
        res.append(tmp)
    return numpy.asarray(res)

def SimpleArrayHistogramEqualization(array, min, max):
    flattened_array = array.flatten()
    sorted_array = numpy.sort(flattened_array)
    cdf = sorted_array.cumsum() / sorted_array.max()
    current_min = min if min is not None else flattened_array.min()
    current_max = max if max is not None else flattened_array.max()
    NormalizeInplace(cdf, imin=current_min, imax=current_max)
    y = numpy.interp(flattened_array, sorted_array, cdf)
    new_array = y.reshape(array.shape)
    return new_array


def histogramEqualization(ndArray, min=None, max=None):
    assert isinstance(ndArray, numpy.ndarray) and len(ndArray.shape) == 2
    res = []
    for row in ndArray.astype(float):
        new_row = SimpleArrayHistogramEqualization(row, min, max)
        assert all([min <= val <= max for val in new_row])
    	res.append(new_row)
    return numpy.asarray(res)


def get_max_rel_error(v1, v2):
	m=np.max(np.abs(v1 - v2))
	n=np.argmax(np.abs(v1-v2))

	L = []
	if v1[n] != 0:
		L.append(v1[n])
	if v2[n] != 0:
		L.append(v2[n])
	if len(L) == 0:
		L = [0]
	return max(L)



def safe_exp(x):
	if np.any(x > 500):
		print "Warn : overflow of exp... exp(%f)" % (np.max(x),)
		x = (np.minimum(x, 500))
	if np.any(x < -500):
		print "Warn : underflow of exp but this is ok : exp(%f)" % (np.min(x),)
	return np.exp(x)


def safe_log(x):
 	if np.any(x == 0):
		print "Warn:Zero value for log"
		x = (np.maximum(x, 1e-300) )
	if np.any(x) < 0:
		print "Err:negative value for log : exp(%f)" % (np.min(x),)
		lumberstack()
		x = np.maximum(x, 0)
	return np.log(x)


from numpy import dot, sqrt, diag
from numpy.linalg import eigh
def whiten(X,fudge=1E-10):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d,V = eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = diag(1./sqrt(d+fudge))

   # whitening matrix
   W = dot(dot(V,D),V.T)

   # multiply by the whitening matrix
   X = dot(X,W)
   return X


def lumberstack():
	traceback.print_stack()
	print repr(traceback.format_stack())

def pad(from_array, coords, coords_next, padding):
	return np.lib.pad(from_array, [ ( (t-f) - s, 0) for f, t, s in zip(coords, coords_next, from_array.shape)], padding)

def pad_to_desired_shape(from_array, desired_shape, padding):

	return np.lib.pad(from_array,  [ ((d-s), 0) for s, d in zip(from_array.shape, desired_shape)], padding)

def image3d_to_scatter(data):
	X = []
	Y = []
	Z = []
	V = []
	for z in xrange(data.shape[0]):
		for y in xrange(data.shape[1]):
			for x in xrange(data.shape[2]):
				v = data[z][y][x]
				if v != 0:
					X.append(x)
					Y.append(y)
					Z.append(z)
					V.append(v)
	return X, Y, Z, V


def export2arff(inputs, targets, targets_names, filename, relationName, output_folder):
      
        print '%s/%s' % (output_folder, filename)
        file = open('%s/%s' % (output_folder, filename), 'w')
        s = '@RELATION %s\n\n' % relationName
        for i in xrange(inputs.shape[1]):
            s += '@ATTRIBUTE value_%d NUMERIC\n' % i
        s += '@ATTRIBUTE class {'
        for name in targets_names:
            s += name + ','
        if s[-1] == ',' : s = s[:-1]
        s += '}\n\n'
        s += '@DATA\n'
        for i in xrange(inputs.shape[0]):
            for j in xrange(inputs.shape[1]):
                s += str(inputs[i,j]) + ','
            s += targets_names[targets[i, -1]] + '\n'
        if s[-1] == '\n' : s = s[:-1]
        file.write(s)
        file.close()


from md5 import md5
def ndhash(ndarray):
    return md5("".join(map(str, ndarray.flatten()))).hexdigest()


def concat_features(features):
    nb_examples = features[features.keys()[0]].shape[0]
    data = []
    for i in xrange(nb_examples):
        D = []
        for d in features.values():
            D.extend((d[i].flatten()))
        data.append(D)
    return data

def translate(data, nb, axes):
    for a in axes:
        data = np.roll(data, nb[a], axis=a)
    for a in axes:
        slices = [slice(0, el) if i != a else slice(0, nb[a]) for i, el in enumerate(data.shape)]
        data[slices] = 0
    return data

def crop_data(data, crop_wanted):
    best = 0
    best_id = 0, 0, 0
    for z in xrange(30 - crop_wanted[0] + 1):
        for y in xrange(18 - crop_wanted[1] + 1):
            for x in xrange(18 - crop_wanted[2] + 1):
                amount = np.sum(data[z:z+crop_wanted[0], y:y+crop_wanted[1] , x:x+crop_wanted[2]])
                if amount > best:
                    best_id = z, y, x
                    best = amount
    return data[best_id[0]:best_id[0]+crop_wanted[0], best_id[1]:best_id[1]+crop_wanted[1], best_id[2]:best_id[2]+crop_wanted[2]]


def histo_equal(data):
    L = 256

    def apply_T(T):

        def apply(x):
            return round(T[x])
        return np.vectorize(apply)

    for i in xrange(data.shape[0]):
        image = data[i]
        nb_pixels = np.prod(image.shape)
        image = image.reshape((nb_pixels,))
        occ = defaultdict(int)
        for value in image:
            occ[value] += 1
        T = {}
        n = 0

        value_min = min(occ.keys())
        value_max = max(occ.keys())
        for val in range(0, L):
            n += occ[val]
            T[val] = (float(n - value_min) / (nb_pixels - value_min)) * (L - 1)
        transform = apply_T(T)

        data[i] = transform(data[i])

    return data


def to_hamming(targets):
    if type(targets) == np.ndarray and len(targets.shape) > 1:
        targets = targets[:, 0]

    t = sorted(set(tuple(targets)))
    mapping = {}
    n = 0.5
    M = [-n] * len(t)
    for i, m in enumerate(t):
        M[i] = n
        mapping[m] = copy.copy(M)
        M[i] = -n
    return np.array([mapping[t] for t in targets])

def from_hamming(targets):
    return np.argmax(targets, axis=1)


def import_object(name):
    components = name.split('.')
    mod = __import__(".".join(components[0:-1]))
    return getattr(mod, components[-1])

def save_learner(learner, filename):
    fd = gzip.open(ressource_full_filename(with_job_suffix(filename), LEARNERS, output=True), "w")
    learner.save(fd)
    fd.close()

def conv_2d_filter(img, filter_):
    new_img = np.zeros(img.shape)
    for y in xrange(img.shape[0] - filter_.shape[0] + 1):
        for x in xrange(img.shape[1] - filter_.shape[1] + 1):
            new_img[y:y + filter_.shape[0], x:x + filter_.shape[1]] +=  np.sum(img[y:y + filter_.shape[0], x:x + filter_.shape[1]] * filter_)

    return new_img


def conv_3d_filter(img, filter_, stride=(1, 1, 1)):
    new_img = np.zeros(img.shape)
    for z in xrange( (img.shape[0] - filter_.shape[0]) / stride[0] + 1):
        for y in xrange( (img.shape[1] - filter_.shape[1]) / stride[1] + 1):
            for x in xrange(  (img.shape[2] - filter_.shape[2]) / stride[2] + 1):
                new_img[z*stride[0]:z*stride[0]+filter_.shape[0], y*stride[1]:y*stride[1] + filter_.shape[1], x*stride[2]:x*stride[2] + filter_.shape[2]] +=  np.sum(img[z*stride[0]:z*stride[0]+filter_.shape[0], y*stride[1]:y*stride[1] + filter_.shape[1], x*stride[2]:x*stride[2] + filter_.shape[2]] * filter_)

    return new_img

