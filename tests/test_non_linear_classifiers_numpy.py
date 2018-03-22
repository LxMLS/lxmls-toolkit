import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)
import numpy as np

import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.utils import AmazonData
from lxmls.deep_learning.utils import Model, glorot_weight_init, index2onehot, logsumexp
from lxmls.deep_learning.numpy_models.mlp import NumpyMLP
from lxmls.deep_learning.mlp import get_mlp_parameter_handlers, get_mlp_loss_range

tolerance = 2 #TODO #FIXME Implementations give random results in a +-1 margin.
              # Check the source of this randomness

@pytest.fixture(scope='module')
def corpus():
    return srs.SentimentCorpus("books")

@pytest.fixture(scope='module')
def data(corpus):
    return AmazonData(corpus=corpus)

# exercise 1
def test_numpy_log_linear(corpus, data):

    class NumpyLogLinear(Model):

        def __init__(self, **config):

            # Initialize parameters
            weight_shape = (config['input_size'], config['num_classes'])
            # after Xavier Glorot et al
            self.weight = glorot_weight_init(weight_shape, 'softmax')
            self.bias = np.zeros((1, config['num_classes']))
            self.learning_rate = config['learning_rate']

        def log_forward(self, input=None):
            """Forward pass of the computation graph"""

            # Linear transformation
            z = np.dot(input, self.weight.T) + self.bias

            # Softmax implemented in log domain
            log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

            return log_tilde_z

        def predict(self, input=None):
            """Prediction: most probable class index"""
            return np.argmax(np.exp(self.log_forward(input)), axis=1)

        def update(self, input=None, output=None):
            """Stochastic Gradient Descent update"""

            # Probabilities of each class
            class_probabilities = np.exp(self.log_forward(input))
            batch_size, num_classes = class_probabilities.shape

            # Error derivative at softmax layer
            I = index2onehot(output, num_classes)
            error = (class_probabilities - I) / batch_size

            # Weight gradient
            gradient_weight = np.zeros(self.weight.shape)
            for l in range(batch_size):
                gradient_weight += np.outer(error[l, :], input[l, :])

            # Bias gradient
            gradient_bias = np.sum(error, axis=0, keepdims=True)

            # SGD update
            self.weight = self.weight - self.learning_rate * gradient_weight
            self.bias = self.bias - self.learning_rate * gradient_bias

    learning_rate = 0.05
    model = NumpyLogLinear(
        input_size=corpus.nr_features,
        num_classes=2,
        learning_rate=learning_rate
    )

    # Define number of epochs and batch size
    num_epochs = 10
    batch_size = 30

    # Get batch iterators for train and test
    train_batches = data.batches('train', batch_size=batch_size)
    test_set = data.batches('test', batch_size=None)[0]

    # Get intial accuracy
    hat_y = model.predict(input=test_set['input'])
    accuracy = 100*np.mean(hat_y == test_set['output'])
    assert np.allclose(accuracy, 51.25, tolerance)

    for epoch in range(num_epochs):
        for batch in train_batches:
            model.update(input=batch['input'], output=batch['output'])
        hat_y = model.predict(input=test_set['input'])
        accuracy = 100*np.mean(hat_y == test_set['output'])

    assert np.allclose(accuracy, 81.75, tolerance)


# exercise 2
def test_backpropagation_numpy(corpus, data):

    # Model
    geometry = [corpus.nr_features, 20, 2]
    activation_functions = ['sigmoid', 'softmax']

    # Optimization
    learning_rate = 0.05
    num_epochs = 10
    batch_size = 30

    # Model
    geometry = [corpus.nr_features, 20, 2]
    activation_functions = ['sigmoid', 'softmax']

    model = NumpyMLP(
        geometry=geometry,
        activation_functions=activation_functions,
        learning_rate=learning_rate
    )

    # Get functions to get and set values of a particular weight of the model
    get_parameter, set_parameter = get_mlp_parameter_handlers(
        layer_index=1,
        is_bias=False,
        row=0,
        column=0
    )

    # Get batch of data
    batch = data.batches('train', batch_size=batch_size)[0]

    # Get loss and weight value
    current_loss = model.cross_entropy_loss(batch['input'], batch['output'])
    current_weight = get_parameter(model.parameters)

    # Get range of values of the weight and loss around current parameters values
    weight_range, loss_range = get_mlp_loss_range(model, get_parameter, set_parameter, batch)

    # Get the gradient value for that weight
    gradients = model.backpropagation(batch['input'], batch['output'])
    current_gradient = get_parameter(gradients)

    # Get batch iterators for train and test
    train_batches = data.batches('train', batch_size=batch_size)
    test_set = data.batches('test', batch_size=None)[0]

    # Epoch loop
    for epoch in range(num_epochs):

        # Batch loop
        for batch in train_batches:
            model.update(input=batch['input'], output=batch['output'])

        # Prediction for this epoch
        hat_y = model.predict(input=test_set['input'])

        # Evaluation
        accuracy = 100*np.mean(hat_y == test_set['output'])

        # Inform user
        print("Epoch %d: accuracy %2.2f %%" % (epoch+1, accuracy))

    assert np.allclose(accuracy, 80.25, tolerance)

