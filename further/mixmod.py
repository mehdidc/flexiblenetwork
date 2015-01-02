

from ILC import load_data
from show import show_3d
import sys
import numpy as np
from sklearn import mixture

def mixmod(data, targets):
    a, b = np.random.randint(0, len(data)-1), np.random.randint(0, len(data)-1)
    rec1 = np.zeros((30, 18, 18))
    rec2 = np.zeros((30, 18, 18))
    mix = data[a] +  data[b]
    diff = data[a] - data[b]
    D = np.tanh(diff)
    show_3d( [D]  , [""])
    #show_3d( [data[a]]  , ["a"])
    #show_3d( [data[b]]  , ["b"])
    
    for y in xrange(18):
        for x in xrange(18):
            D1 = data[a][:, x, y]
            D2 = data[b][:, x, y]
            S = np.c_[D1, D2]
            
            if np.sum(D1) == 0:
                continue
            if np.sum(D2) == 0:
                continue
            mat = np.array([ [1, 4], [4, 1] ])
            X = np.dot(S, mat.T)
            ica = FastICA(n_components=2, max_iter=1000, tol=0.001, whiten=True)
            try:
                S_ = ica.fit_transform(X)
                rec1[:, x, y] = S[:, 0]
                rec2[:, x, y] = S[:, 1]
            except:
                print "prob"
                pass

    """
    clf = mixture.GMM(n_components=2, covariance_type='full', n_iter=1000, n_init=10, thresh=0.000001)
    points = []
    for z in xrange(30):
        for y in xrange(18):
            for x in xrange(18):
                if D[z, y, x]>0:
                    e = D[z, y, x]
                    e_ = mix[z, y, x]
                    s = (e_-mix.min())/(mix.max()-mix.min())
                    for i in xrange(10):
                        t1 = np.random.normal(0, s) 
                        t2 = np.random.normal(0, s)
                        t3 = np.random.normal(0, s)
                        points.append( [x+t1, y+t2, z+t3] )
    clf.fit(points)
    D_rec = np.zeros(D.shape)
    for p, score in zip(points, clf.score_samples(points)[1]):
        x, y, z = p
        D_rec[z, y, x] = score[0]
    show_3d([D_rec], ["mixture of a and b"])
    """
    rec1 = (rec1 - rec1.mean()) / rec1.std()
    rec2 = (rec2 - rec2.mean()) / rec2.std()
    rec1[rec1<5] = 0
    rec2[rec2<5]=0
    show_3d([rec1],["a"])
    show_3d([rec2], ["b"]) 

def gaussian(data, targets):
    gmm = mixture.GMM( n_components=1, covariance_type='diag', random_state=None, n_iter=10, n_init=5, params='wmc', init_params='wmc')
    data = data.reshape( (data.shape[0], 30*18*18) )
    gmm.fit(data)
    samples = gmm.sample(10)
    samples[samples< 5]  = 0
    show_3d(samples.reshape( (samples.shape[0], 30, 18, 18)), ["gibbs"]*samples.shape[0]) 
from sklearn.decomposition import FastICA, PCA

if __name__ == "__main__":
    file = sys.argv[1]
    data, targets = load_data(file)
    mixmod(data, targets)
