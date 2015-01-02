import numpy as np

def list_with_one_in(size, idx_one):
    l = [0] * size
    l[idx_one] = 1
    return l
import sys
class Dataset(object):


    def __init__(self, input_, output, check=True):

        if not isinstance(input_, np.ndarray):
            input_ = np.array(input_)
        if not isinstance(output, np.ndarray):
            output = np.array(output)

        if check==True:
            assert input_.shape[0] == output.shape[0]
        self.input = input_
        self.output = output
        self.nb_examples = self.input.shape[0]
    
    def truncate(self, nb):
        self.input = self.input[0:nb]
        self.output = self.output[0:nb]
        self.nb_examples = nb
    def shuffle(self):
        order = range(self.nb_examples)
        np.random.shuffle(order)
        self.apply_order(order)

    def groupby_output(self, key=lambda x:x):
        order = range(self.nb_examples)
        order = sorted(order, key=lambda i:key(self.output[i]))
        self.apply_order(order)

    def apply_order(self, order):
        self.input = np.array([self.input[order[i]] for i in xrange(self.nb_examples)])
        self.output = np.array([self.output[order[i]] for i in xrange(self.nb_examples)])
    
    def normalize_input_mean_std(self):
        self.transform_input_whole(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0))

    def normalize_input_min_max(self):
        self.transform_input_whole(lambda x: (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)))

    def transform_input_each(self, T):
        self.input = np.array(map(T, self.input))

    def transform_output_from_classid_to_binvectors(self, nbclasses=None):
        if nbclasses is None:
            nbclasses = len(set(self.output.flatten()))
        self.transform_output_each(lambda x: np.array(list_with_one_in(nbclasses, x)))

    def transform_output_each(self, T):
        self.output = np.array(map(T, self.output))

    def transform_input_whole(self, T):
        self.input = T(self.input)

    def transform_output_whole(self, T):
        self.output = T(self.output)
    
    def break_to_datasets(self, ratios):
        assert sum(ratios) == 1
        datasets = []
        idx = 0

        for ratio in ratios:
            nb = int(ratio*self.nb_examples)
            from_, to = idx, idx + nb
            to = min(to, self.nb_examples)
            dataset = Dataset(self.input[from_:to], self.output[from_:to])
            datasets.append(dataset)
            idx += nb
        return datasets

    def get_random_replacement_batch(self, batch_size):
        els = [np.random.randint(0, self.nb_examples) for j in xrange(batch_size)]
        inputs = np.array([self.input[el] for el in els])
        outputs = np.array([self.output[el] for el in els])
        return inputs, outputs

    def get_random_batch(self, batch_size):
        els = np.random.choice(range(self.input.shape[0]), size=batch_size, replace=False)
        inputs = np.array([self.input[el] for el in els])
        outputs = np.array([self.output[el] for el in els])
        return inputs, outputs

    def get_random_batch_as_dataset(self, batch_size):
        return Dataset(*get_random_batch(batch_size))

    def get_random_replacement_batch_as_dataset(self, batch_size):
        return Dataset(*self.get_random_replacement_batch(batch_size))

    def get_random_batch(self, batch_size):
        nb_batches = self.nb_examples / batch_size + (1 if (self.nb_examples % batch_size != 0) else 0)
        batch_id = np.random.randint(0, nb_batches)

        from_, to = batch_id*batch_size, min(self.nb_examples, (batch_id+1)*batch_size)
        inputs = self.input[from_:to]
        outputs = self.output[from_:to]
        return inputs, outputs

    def get_random_batch_as_dataset(self, batch_size):
        return Dataset(*self.get_random_batch(batch_size))

