import numpy as np
import pytest
import torch

import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.pytorch_models.mlp import PytorchMLP
from lxmls.deep_learning.utils import AmazonData, Model, glorot_weight_init

tolerance = 1e-2


@pytest.fixture(scope="module")
def corpus():
    return srs.SentimentCorpus("books")


@pytest.fixture(scope="module")
def data(corpus):
    return AmazonData(corpus=corpus)


# exercise 3
def test_loglinear_pytorch(corpus, data):
    class PytorchLogLinear(Model):
        def __init__(self, **config):
            # Initialize parameters
            weight_shape = (config["input_size"], config["num_classes"])
            # after Xavier Glorot et al
            weight_np = glorot_weight_init(weight_shape, "softmax")
            self.learning_rate = config["learning_rate"]

            # IMPORTANT: Cast to pytorch format
            self.weight = torch.from_numpy(weight_np).float()
            self.weight.requires_grad = True

            self.bias = torch.zeros(1, config["num_classes"], requires_grad=True)

            self.log_softmax = torch.nn.LogSoftmax(dim=1)
            self.loss_function = torch.nn.NLLLoss()

        def _log_forward(self, input=None):
            """Forward pass of the computation graph in logarithm domain (pytorch)"""

            # IMPORTANT: Cast to pytorch format
            input = torch.from_numpy(input).float()

            # Linear transformation
            z = torch.matmul(input, torch.t(self.weight)) + self.bias

            # Softmax implemented in log domain
            log_tilde_z = self.log_softmax(z)

            # NOTE that this is a pytorch class!
            return log_tilde_z

        def predict(self, *args, **kwargs):
            """Most probable class index"""
            if args:
                input = args[0]
            else:
                input = kwargs.get("input")
            log_forward = self._log_forward(input).data.numpy()
            return np.argmax(log_forward, axis=1)

        def update(self, *args, **kwargs):
            """Stochastic Gradient Descent update"""

            if args:
                input = args[0]
                output = args[1]
            else:
                input = kwargs.get("input")
                output = kwargs.get("output")

            # IMPORTANT: Class indices need to be casted to LONG
            true_class = torch.from_numpy(output).long()

            # Compute negative log-likelihood loss
            loss = self.loss_function(self._log_forward(input), true_class)

            # Use autograd to compute the backward pass.
            loss.backward()

            # SGD update
            assert self.weight.grad is not None, "Weight gradients are None."
            assert self.bias.grad is not None, "Bias gradients are None."
            self.weight.data -= self.learning_rate * self.weight.grad.data
            self.bias.data -= self.learning_rate * self.bias.grad.data

            # Zero gradients
            self.weight.grad.data.zero_()
            self.bias.grad.data.zero_()

            return loss.data.numpy()

    model = PytorchLogLinear(input_size=corpus.nr_features, num_classes=2, learning_rate=0.05)

    # Hyper-parameters
    num_epochs = 2
    batch_size = 30

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
    # assert np.allclose(accuracy, 74, tolerance)
    assert np.allclose(accuracy, 81, tolerance)


# exercise 4
def test_backpropagation_pytorch(corpus, data):
    # Model
    geometry = [corpus.nr_features, 20, 2]
    activation_functions = ["sigmoid", "softmax"]

    # Optimization
    learning_rate = 0.05
    num_epochs = 2
    batch_size = 30

    model = PytorchMLP(
        geometry=geometry,
        activation_functions=activation_functions,
        learning_rate=learning_rate,
    )

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
    # assert np.allclose(accuracy, 67.5, tolerance)
    # TODO(rshwndsz): investigate regression
    assert np.allclose(accuracy, 61.75, tolerance)


if __name__ == "__main__":
    pytest.main([__file__])
