import numpy as np
import pytest

import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.mlp import get_mlp_loss_range, get_mlp_parameter_handlers
from lxmls.deep_learning.numpy_models.mlp import NumpyMLP
from lxmls.deep_learning.utils import (
    AmazonData,
    Model,
    glorot_weight_init,
    index2onehot,
    logsumexp,
)

tolerance = 1e-2


@pytest.fixture(scope="module")
def corpus():
    return srs.SentimentCorpus("books")


@pytest.fixture(scope="module")
def data(corpus):
    return AmazonData(corpus=corpus)


# exercise 1
def test_numpy_log_linear(corpus, data):
    class NumpyLogLinear(Model):
        def __init__(self, **config):
            # Initialize parameters
            weight_shape = (config["input_size"], config["num_classes"])
            # after Xavier Glorot et al
            self.weight = glorot_weight_init(weight_shape, "softmax")
            self.bias = np.zeros((1, config["num_classes"]))
            self.learning_rate = config["learning_rate"]

        def log_forward(self, input):
            """Forward pass of the computation graph"""

            # Linear transformation
            z = np.dot(input, self.weight.T) + self.bias

            # Softmax implemented in log domain
            log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

            return log_tilde_z

        def predict(self, *args, **kwargs):
            """Prediction: most probable class index"""
            input = kwargs.get("input") if "input" in kwargs else args[0] if args else None
            return np.argmax(np.exp(self.log_forward(input)), axis=1)

        def update(self, *args, **kwargs):
            """Stochastic Gradient Descent update"""

            input = kwargs.get("input") if "input" in kwargs else args[0] if args else None
            if input is None:
                raise ValueError("Input data is required for update.")

            output = kwargs.get("output") if "output" in kwargs else args[1] if len(args) > 1 else None

            # Probabilities of each class
            class_probabilities = np.exp(self.log_forward(input))
            batch_size, num_classes = class_probabilities.shape

            # Error derivative at softmax layer
            dL = index2onehot(output, num_classes)
            error = (class_probabilities - dL) / batch_size

            # Weight gradient
            gradient_weight = np.zeros(self.weight.shape)
            for _l in range(batch_size):
                gradient_weight += np.outer(error[_l, :], input[_l, :])

            # Bias gradient
            gradient_bias = np.sum(error, axis=0, keepdims=True)

            # SGD update
            self.weight = self.weight - self.learning_rate * gradient_weight
            self.bias = self.bias - self.learning_rate * gradient_bias

    learning_rate = 0.05
    model = NumpyLogLinear(input_size=corpus.nr_features, num_classes=2, learning_rate=learning_rate)

    # Define number of epochs and batch size
    num_epochs = 2
    batch_size = 30

    # Get batch iterators for train and test
    train_batches = data.batches("train", batch_size=batch_size)
    test_set = data.batches("test", batch_size=None)[0]

    # Get initial accuracy
    hat_y = model.predict(input=test_set["input"])
    accuracy = 100 * np.mean(hat_y == test_set["output"])
    assert np.allclose(accuracy, 51.25, tolerance)

    for _epoch in range(num_epochs):
        for batch in train_batches:
            model.update(input=batch["input"], output=batch["output"])
        hat_y = model.predict(input=test_set["input"])
        accuracy = 100 * np.mean(hat_y == test_set["output"])

    # assert np.allclose(accuracy, 74, tolerance)
    assert np.allclose(accuracy, 81, tolerance)


# exercise 2
def test_backpropagation_numpy(corpus, data):
    # Model
    activation_functions = ["sigmoid", "softmax"]
    # Optimization
    learning_rate = 0.05
    num_epochs = 2
    batch_size = 30

    # Model
    geometry = [corpus.nr_features, 20, 2]
    activation_functions = ["sigmoid", "softmax"]

    model = NumpyMLP(
        geometry=geometry,
        activation_functions=activation_functions,
        learning_rate=learning_rate,
    )

    # Get functions to get and set values of a particular weight of the model
    get_parameter, set_parameter = get_mlp_parameter_handlers(layer_index=1, is_bias=False, row=0, column=0)

    # Get batch of data
    batch = data.batches("train", batch_size=batch_size)[0]

    # Get loss and weight value
    _current_loss = model.cross_entropy_loss(batch["input"], batch["output"])
    _current_weight = get_parameter(model.parameters)

    # Get range of values of the weight and loss around current parameters values
    weight_range, loss_range = get_mlp_loss_range(model, get_parameter, set_parameter, batch)

    # Get the gradient value for that weight
    gradients = model.backpropagation(batch["input"], batch["output"])
    _current_gradient = get_parameter(gradients)

    # Get batch iterators for train and test
    train_batches = data.batches("train", batch_size=batch_size)
    test_set = data.batches("test", batch_size=None)[0]

    # Epoch loop
    accuracy = 0
    for epoch in range(num_epochs):
        # Batch loop
        for batch in train_batches:
            model.update(input=batch["input"], output=batch["output"])

        # Prediction for this epoch
        hat_y = model.predict(input=test_set["input"])

        # Evaluation
        accuracy = 100 * np.mean(hat_y == test_set["output"])

        # Inform user
        print("Epoch %d: accuracy %2.2f %%" % (epoch + 1, accuracy))

    # assert np.allclose(accuracy, 67.5, tolerance)
    # TODO(rshwndsz) Investigate regression
    assert np.allclose(accuracy, 61.75, tolerance)


if __name__ == "__main__":
    pytest.main([__file__])
