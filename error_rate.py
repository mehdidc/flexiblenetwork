import numpy as np

def classification_error_rate(real, predicted):
    nb = 0
    for err, which_class in where_error(real, predicted):
        nb += 1
    return float(nb) / real.shape[0]

def regression_error(real, predicted):
    return np.mean((real - predicted)**2)

def confusion_matrix(real, predicted):
    assert real.shape[0] == predicted.shape[0]
    nb_classes = real.shape[1]
    if nb_classes == 1:
        nb_classes = 2        
        real = np.array([([1, -1] if real[i]>=0 else [-1, 1])  for i in xrange(real.shape[0])])
        predicted = np.array([([1, -1] if predicted[i]>=0 else [-1, 1])  for i in xrange(predicted.shape[0])])

    confmat = np.zeros((nb_classes, nb_classes))
    real = to_binary(real)
    predicted = to_binary(predicted)
    for i in xrange(nb_classes):
        for j in xrange(nb_classes):
            confmat[i, j] = np.sum((real[:, i]==1) * (predicted[:, j]==1))
    return confmat.astype(np.int)

def to_binary(v):
    _, res = np.indices(v.shape)
    return (res == np.argmax(v, axis=1)[:, np.newaxis])

def where_error(real, predicted):
    assert real.shape[0] == predicted.shape[0]
    
    nb_classes = real.shape[1]
    if nb_classes == 1:
        nb_classes = 2        
        real = np.array([([1, -1] if real[i]>=0 else [-1, 1])  for i in xrange(real.shape[0])])
        predicted = np.array([([1, -1] if predicted[i]>=0 else [-1, 1])  for i in xrange(predicted.shape[0])])

    nb_examples = real.shape[0]
    err = 0
    nb = real.shape[0]
    for i in xrange(nb_examples):
        if np.argmax(real[i]) != np.argmax(predicted[i]):
            yield i, np.argmax(real[i])
