from ILC import load_data
#from show import show_3d
import sys
import numpy as np
from feature_map import FeatureMap, Learner, Layer
from connections import ConnectorAll
import config
import random
from pylab import plot, show, draw
from collections import deque


import matplotlib.pyplot as plt

def learn(data, targets):
    nbhidden = (100,)
    nbhidden2 = (200,)
    nbinput = (5, 5)
    input_fmap  = FeatureMap( nbinput )

    hidden_fmap = FeatureMap (nbhidden)
    ConnectorAll( input_fmap, hidden_fmap ).connect()
    
    hidden_fmap2 = FeatureMap( nbhidden2)
    ConnectorAll( hidden_fmap, hidden_fmap2).connect()

    output_fmap = FeatureMap( (1,), activation_func_name="tanh" )
    ConnectorAll( hidden_fmap2, output_fmap ).connect()

    options = config.options["LEARNERS"][config.options["LEARNER"]]
    learner = Learner([Layer([input_fmap]), Layer([hidden_fmap]), Layer([hidden_fmap2]), Layer([output_fmap])], options=options)


    targets = ["elastic" if t[0] == 1 else "inelastic" for t in np.array(targets)]
    data = np.array([d for d, l in zip(data, targets) if l=="elastic"])
    
    points = []
    for epoch in xrange(1000):
        print "epoch %d" % (epoch,)
        for example in [data[0]]:
            error = 0
            nberrs = 0
            for z in range(1, 29):
                example_z = np.pad(example[z], nbinput, "constant")
                for y in xrange(18):
                    for x in xrange(18):
                        
                        d = example_z[y + nbinput[0]:y+2*nbinput[0], x+nbinput[1]:x+2*nbinput[1]]
                        #d[d.shape[0]/2, d.shape[1]/2]
                        o = example[z + 1, y, x]
                        inp = {input_fmap : np.array([d])}
                        outp = {output_fmap: np.array([o])}
                        pred = learner.predict(inp)[output_fmap][0][0]
                        err= ((pred-o)**2) / (o**2 + (1 if o==0 else 0))
                        error +=err
                        learner.train( inp, outp )
                        nberrs += 1
            points.append(float(error) / nberrs)
            if epoch % 10==0:
                plot(range(len(points)), list(points))
                show()

if __name__ == "__main__":
    file = sys.argv[1]
    data, targets = load_data(file)
    learn(data, targets)
