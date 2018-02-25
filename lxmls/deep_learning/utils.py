import numpy as np

#
# UTILITIES
#


def logsumexp(a, axis=None, keepdims=False):
    """
    This is an improvement over the original logsumexp of
    scipy/maxentropy/maxentutils.py that allows specifying an axis to sum
    It also allows keepdims=True.
    """
    if axis is None:
        a = np.asarray(a)
        a_max = a.max()
        return a_max + np.log(np.exp(a-a_max).sum())
    else:
        a_max = np.amax(a, axis=axis, keepdims=keepdims)
        return a_max + np.log((np.exp(a-a_max)).sum(axis, keepdims=keepdims))


def index2onehot(index, N):
    """
    Transforms index to one-hot representation, for example

    Input: e.g. index = [1, 2, 0], N = 4
    Output:     [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    """
    L = index.shape[0]
    onehot = np.zeros((L, N))
    for l in np.arange(L):
        onehot[l, index[l]] = 1
    return onehot


def glorot_weight_init(shape, activation_function, random_seed=None):
    """Layer weight initialization after Xavier Glorot et. al"""

    if random_seed is None:
        random_seed = np.random.RandomState(1234)

    # Weights are uniform distributed with span depending on input and output
    # sizes
    num_inputs, num_outputs = shape
    weight = random_seed.uniform(
        low=-np.sqrt(6. / (num_inputs + num_outputs)),
        high=np.sqrt(6. / (num_inputs + num_outputs)),
        size=(num_outputs, num_inputs)
    )

    # Scaling factor depending on non-linearity
    if activation_function == 'sigmoid':
        weight *= 4
    elif activation_function == 'softmax':
        weight *= 4

    return weight

#
# Model and Data
#


class AmazonData(object):
    """
    Template
    """
    def __init__(self, **config):

        # Data-sets
        self.datasets = {
            'train': {
                'input': config['corpus'].train_X,
                'output': config['corpus'].train_y[:, 0]
            },
            #  'dev': (config['corpus'].dev_X, config['corpus'].dev_y[:, 0]),
            'test': {
                'input': config['corpus'].test_X,
                'output': config['corpus'].test_y[:, 0]
            }
        }
        # Config
        self.config = config
        # Number of samples
        self.nr_samples = {
           sset: content['output'].shape[0]
           for sset, content in self.datasets.items()
        }

    def size(self, set_name):
        return self.nr_samples[set_name]

    def batches(self, set_name, batch_size=None):

        dset = self.datasets[set_name]
        nr_examples = self.nr_samples[set_name]
        if batch_size is None:
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples*1./batch_size))

        data = []
        for batch_n in range(nr_batch):
            # Colect data for this batch
            data_batch = {}
            for side in ['input', 'output']:
                data_batch[side] = dset[side][
                   batch_n * batch_size:(batch_n + 1) * batch_size
                ]
            data.append(data_batch)

        return DataIterator(data, nr_samples=self.nr_samples[set_name])


class DataIterator(object):
    """
    Basic data iterator
    """

    def __init__(self, data, nr_samples):
        self.data = data
        self.nr_samples = nr_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Model(object):
    def __init__(self, **config):
        self.initialized = False

    def initialize_features(self, *args):
        self.initialized = True
        raise NotImplementedError(
            "Need to implement initialize_features method"
        )

    def get_features(self, input=None, output=None):
        """
        Default feature extraction is do nothing
        """
        return {'input': input, 'output': output}

    def predict(self, *args):
        raise NotImplementedError("Need to implement predict method")

    def update(self, *args):
        # This needs to return at least {'cost' : 0}
        raise NotImplementedError("Need to implement update method")
        return {'cost': None}

    def set(self, **kwargs):
        raise NotImplementedError("Need to implement set method")

    def get(self, name):
        raise NotImplementedError("Need to implement get method")

    def save(self):
        raise NotImplementedError("Need to implement save method")

    def load(self, model_folder):
        raise NotImplementedError("Need to implement load method")
