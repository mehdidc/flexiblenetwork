import numpy as np
from error_rate import classification_error_rate, confusion_matrix, where_error

class Problem(object):
    
    def get_adapted_dataset(self, data, targets):
        pass

    def get_error_rate(self, real, predicted):
        pass

    def get_confusion_matrix(self, real, predicted):
        pass


def values_except_one(nb, defaultvalue, onevalue, oneindex):
    t = [defaultvalue] * nb
    t[oneindex] = onevalue
    return t

class ClassificationProblem(Problem):
   
    def get_adapted_dataset(self, data, targets):
        labels = list(set(targets))
        targets = np.array([values_except_one(len(labels), -1, 1, labels.index(t)) for t in targets])
        return data, targets
    
    def get_error_rate(self, real, predicted):
        return classification_error_rate(real, predicted)

    def get_confusion_matrix(self, real, predicted):
        return confusion_matrix(real, predicted)



class OrdinalRegressionProblem(Problem):
    def __init__(self):
        self.centers = None
        self.thresholds = None

    def get_adapted_dataset(self, data, targets):
        if self.centers is None and self.thresholds is None:
            self.centers, self.thresholds = self.__get_centers_and_thresholds(data, targets)
            print self.centers, self.thresholds

        return data, np.array([ [self.centers[int(t)]] for t in targets])

    def __get_centers_and_thresholds(self, data, targets):
        targets = map(int, targets)
        hist = np.bincount(targets).astype('float')
        hist /= np.sum(hist)
        cumsum = np.cumsum(hist)
        centers = []
        a = 0
        b = 0
        for h in cumsum:
            b = h
            centers.append( (a + b) / 2)
            a = b
        return np.array(centers) * 2 - 1, np.array(cumsum) * 2 - 1
    
    def get_error_rate(self, real, predicted):
        error = 0
        for real_i, predicted_i in zip(real, predicted):
            a = 0
            b = 0
            for i, h in enumerate(self.thresholds):
                b = h
                if (predicted_i >= a and predicted_i <= b):
                    if i != list(self.centers).index(real_i):
                        error += 1
                        break
                a = b
        return float(error) / real.shape[0] 
