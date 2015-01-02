
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pylab
import os
import os.path
import math
import matplotlib.pyplot as plt

import sys

import math

import pylab
import matplotlib

def show_filters(learner, conv_layer_id):
        conv_layer = learner.layers[conv_layer_id]
        input_conv_layer = learner.layers[conv_layer_id - 1]
        
        ind = 1
        step = 1

        nbz = conv_layer.feature_maps[0].M[input_conv_layer.feature_maps[0]].W.shape[0]
        nbsteps = nbz / step
        for i, conv_fmap in enumerate(conv_layer.feature_maps):
            W = conv_fmap.M[input_conv_layer.feature_maps[0]].W
            #min_, max_ = W.flatten().min(), W.flatten().max()
            filter_im = W / np.sqrt(np.sum(W**2))
            filter_im = filter_im * (filter_im > np.max(filter_im)/2)
            print filter_im.shape
            filter_im.flatten().tofile("filters.txt", ",")
            
            for z in xrange(0, nbz, step):
                plt.subplot(nbsteps*len(conv_layer.feature_maps), 1, ind)
                plt.imshow(filter_im[z], cmap=plt.get_cmap("winter"))  
                ind += 1
       
        plt.show()

def g(x, sigma):
    return 1. / (2 * np.pi * sigma**2) * np.exp( - np.sum(x**2) / (2 * sigma**2))

def gaussian_3d_filter(size, sigma):
    F = np.zeros(size)
    for z in xrange(size[0]):
        for y in xrange(size[1]):
            for x in xrange(size[2]):
                F[z, y, x] = g( np.array([x - size[2]/2, y - size[1]/2, z-size[0]/2]), sigma)
    F /= np.sum(F)
    return F
                

from scipy.signal import fftconvolve


from mpl_toolkits.mplot3d import proj3d

def show_3d(dataset, targets=None, indices=None, cmap_name="winter", fig=None, step=False):
    if indices is None:
        indices = xrange(len(dataset))

    i = 0
    for ind in indices:

        data = np.array(dataset[ind])

        if len(data.shape)==1:
            data = data.reshape( (30, 18, 18))
        if targets is None:
            target = ""
        else:
            target = targets[ind]
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(color='white', linestyle='solid')
        label = pylab.annotate("this", xy=(0, 0), textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        def bt(e):
            label.set_text(E[e.ind[0]])

        fig.canvas.mpl_connect('pick_event', bt)


        X, Y, Z, E = [], [], [], []
        cm = plt.get_cmap(cmap_name)
        for z in xrange(data.shape[0]):
            for y in xrange(data.shape[1]):
                for x in xrange(data.shape[2]):
                    if data[z, y, x] > 0:
                        X.append(x)
                        Z.append(y)
                        Y.append(z)
                        E.append(data[z, y, x])
                        #ax.text(x, y, z, data[z, y, x])

        min_E = min(E)
        max_E = max(E)
        C = [cm( (e-min_E)/(max_E-min_E)) for e in E]
        ax.scatter(X, Y, Z,c=C, marker='o', s=100, picker=True)

        ax.set_xticks(range(0, data.shape[2])) 
        ax.set_yticks(range(0, data.shape[0]))
        ax.set_zticks(range(0, data.shape[1]))
        ax.set_xlim((0, data.shape[2]))
        ax.set_ylim((0, data.shape[0]))
        ax.set_zlim((0, data.shape[1]))
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        #plt.zticks(range(0, 18))
        plt.figtext(0., 0.5, "ind:%d, %s" % (i, target))
        i += 1
        if step:
            plt.show()

if __name__ == "__main__":
    filename = sys.argv[1]
    dataset, targets = np.load(filename)
    indices = None
    if len(sys.argv)>2:
        indices = map(int, list(open(sys.argv[2],"r").readlines()))
    print indices
    show_3d(dataset, targets, indices, step=True)


def ILC_show_filters(data):

    learner = Learner.load(open("LEARNER"))
    learner.workers.close()

    input_fmap = learner.layers[0].feature_maps[0]
    conv_fmap = learner.layers[1].feature_maps[0]
    W = conv_fmap.M[input_fmap].W
    M = conv_fmap.M[input_fmap]

    min_, max_ = W.flatten().min(), W.flatten().max()
    W =  (W - min_) / (max_ - min_)

    from show import show_3d
    D = data[4]

    from scipy.ndimage.filters import convolve
    from scipy.signal import convolve2d, fftconvolve


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils import image3d_to_scatter
    from pylab import savefig
    step = 5
    ind = 1
    cm = 'host'
    nbz = D.shape[0] / step
    for z in xrange(0, D.shape[0], step):
        plt.subplot(nbz, 2, ind)
        plt.imshow(D[z], cmap=plt.get_cmap("winter"))
        ind += 2
    
    ind = 2

    for z in xrange(0, D.shape[0], step):
        R=M.transform(np.array([D[z]]))[0]
        plt.subplot(nbz, 2, ind)
        plt.imshow(R, cmap=plt.get_cmap("winter"))
        ind += 2
    

    plt.show()


