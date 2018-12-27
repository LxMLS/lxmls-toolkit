import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)
import numpy as np

from lxmls.readers.pos_corpus import PostagCorpusData
from lxmls.deep_learning.numpy_models.rnn import NumpyRNN
from lxmls.deep_learning.rnn import get_rnn_parameter_handlers, get_rnn_loss_range
from lxmls.deep_learning.pytorch_models.rnn import PytorchRNN
from lxmls.deep_learning.pytorch_models.rnn import FastPytorchRNN
from lxmls.deep_learning.numpy_models.rnn import NumpyRNN

tolerance = 1e-2

@pytest.fixture(scope='module')
def data(): 
    return PostagCorpusData()

# exercise 1
def test_numpy_rnn(data):

    model = NumpyRNN(
        input_size=data.input_size,
        embedding_size=50,
        hidden_size=20,
        output_size=data.output_size,
        learning_rate=0.1
    )
    shape = [x.shape for x in model.parameters]
    assert np.allclose(shape, [(4786, 50), (20, 50), (20, 20), (12, 20)], tolerance)

    # Get functions to get and set values of a particular weight of the model
    get_parameter, set_parameter = get_rnn_parameter_handlers(
        layer_index=-1,
        row=0, 
        column=0
    )

    # Get batch of data
    batch = data.batches('train', batch_size=1)[0]

    # Get loss and weight value
    current_loss = model.cross_entropy_loss(batch['input'], batch['output'])
    current_weight = get_parameter(model.parameters)

    assert np.allclose(current_loss, 2.2440323240301465, tolerance)
    assert np.allclose(current_weight, 0.6374233957270082, tolerance)

    # Get range of values of the weight and loss around current parameters values
    weight_range, loss_range = get_rnn_loss_range(model, get_parameter, set_parameter, batch)

    # Get the gradient value for that weight
    gradients = model.backpropagation(batch['input'], batch['output'])
    current_gradient = get_parameter(gradients)

    assert np.allclose(current_gradient, -0.2927844936170676, tolerance)

    # Hyper-parameters
    num_epochs = 2

    # Get batch iterators for train and test
    train_batches = data.batches('train', batch_size=1)
    dev_set = data.batches('dev', batch_size=1)
    test_set = data.batches('test', batch_size=1)

    for epoch in range(num_epochs):

        # Batch loop
        for batch in train_batches:
            model.update(input=batch['input'], output=batch['output'])

        # Evaluation dev
        is_hit = []
        for batch in dev_set:
            is_hit.extend(model.predict(input=batch['input']) == batch['output'])
        accuracy = 100*np.mean(is_hit)

    # tested for 2 epochs only
    assert np.allclose(accuracy, 31.81, tolerance)
        
    # Evaluation test
    is_hit = []
    for batch in test_set:
        is_hit.extend(model.predict(input=batch['input']) == batch['output'])
    accuracy = 100*np.mean(is_hit)

    assert np.allclose(accuracy, 30.50, tolerance)

# exercise 2
def test_pytorch_rnn(data):

    model = PytorchRNN(
        input_size=data.input_size,
        embedding_size=50,
        hidden_size=20,
        output_size=data.output_size,
        learning_rate=0.1
    )

    # Get gradients for both models
    batch = data.batches('train', batch_size=1)[0]

    num_epochs = 2

    # Get batch iterators for train and test
    train_batches = data.batches('train', batch_size=1)
    dev_set = data.batches('dev', batch_size=1)
    test_set = data.batches('test', batch_size=1)

    for epoch in range(num_epochs):

        # Batch loop
        for batch in train_batches:
            model.update(input=batch['input'], output=batch['output'])

        # Evaluation dev
        is_hit = []
        for batch in dev_set:
            is_hit.extend(model.predict(input=batch['input']) == batch['output'])
        accuracy = 100*np.mean(is_hit)

    # tested for 2 epochs only
    assert np.allclose(accuracy, 31.81, tolerance)
        
    # Evaluation test
    is_hit = []
    for batch in test_set:
        is_hit.extend(model.predict(input=batch['input']) == batch['output'])
    accuracy = 100*np.mean(is_hit)

    assert np.allclose(accuracy, 30.50, tolerance)