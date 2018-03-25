from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from lxmls.deep_learning.rnn import RNN


def cast_float(variable):
    return Variable(torch.from_numpy(variable).float(), requires_grad=True)


def cast_int(variable):
    return Variable(torch.from_numpy(variable).long(), requires_grad=True)


class PytorchRNN(RNN):
    """
    Basic RNN with forward-pass and gradient computation in Pytorch
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

        # First parameters are the embeddings
        # instantiate the embedding layer first
        self.embedding_layer = torch.nn.Embedding(
            config['input_size'],
            config['embedding_size']
        )

        # Set its value to the stored weight
        self.embedding_layer.weight.data = cast_float(self.parameters[0]).data
        self.parameters[0] = self.embedding_layer.weight

        # Log softmax
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        # Negative-log likelihood
        self.loss = torch.nn.NLLLoss()

        # Need to cast  rest of weights
        num_parameters = len(self.parameters)
        for index in range(1, num_parameters):
            # Get weigths and bias of the layer (even and odd positions)
            self.parameters[index] = cast_float(self.parameters[index])

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self._log_forward(input).data.numpy())
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):
            # Update weight
            self.parameters[m].data -= learning_rate * gradients[m]

    def _log_forward(self, input):
        """
        Forward pass
        """

        # Ensure the type matches torch type
        input = Variable(torch.from_numpy(input).long())

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        embedding_size, vocabulary_size = W_e.shape
        hidden_size = W_h.shape[0]
        nr_steps = input.shape[0]

        # FORWARD PASS COMPUTATION GRAPH

        # ----------
        # Solution to Exercise 2

        raise NotImplementedError("Implement Exercise 2")

        # End of solution to Exercise 2
        # ----------

        return log_p_y

    def backpropagation(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        output = Variable(
            torch.from_numpy(output).long(),
            requires_grad=False
        )

        # Zero gradients
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.data.zero_()

        # Compute negative log-likelihood loss
        log_p_y = self._log_forward(input)
        cost = self.loss(log_p_y, output)
        # Use autograd to compute the backward pass.
        cost.backward()

        num_parameters = len(self.parameters)
        # Update parameters
        gradient_parameters = []
        for index in range(0, num_parameters):
            gradient_parameters.append(self.parameters[index].grad.data)

        return gradient_parameters


class FastPytorchRNN(RNN):
    """
    Basic RNN with forward-pass and gradient computation in Pytorch. Uses native
    Pytorch RNN
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

        # First parameters are the embeddings
        # instantiate the embedding layer first
        self.embedding_layer = torch.nn.Embedding(
            config['input_size'],
            config['embedding_size']
        )
        # Set its value to the stored weight
        self.embedding_layer.weight.data = cast_float(self.parameters[0]).data

        # RNN
        self.rnn = torch.nn.RNN(
            config['embedding_size'],
            config['hidden_size'],
            bias=False
        )
        # TODO: Set paremeters here

        # Log softmax
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

        # Negative-log likelihood
        self.loss = torch.nn.NLLLoss()

        # Get the parameters
        self.parameters = (
            [self.embedding_layer.weight] +
            list(self.rnn.parameters()) +
            [cast_float(self.parameters[-1])]
        )

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self._log_forward(input).data.numpy())
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):
            # Update weight
            self.parameters[m].data -= learning_rate * gradients[m]

    def _log_forward(self, input):
        """
        Forward pass
        """

        # Ensure the type matches torch type
        input = Variable(torch.from_numpy(input).long())

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        embedding_size, vocabulary_size = W_e.shape

        # FORWARD PASS COMPUTATION GRAPH

        # Word Embeddings
        z_e = self.embedding_layer(input)

        # RNN
        h, _ = self.rnn(z_e[:, None, :])

        # Output layer
        y = torch.matmul(h[:, 0, :], torch.t(W_y))

        # Log-Softmax
        log_p_y = self.logsoftmax(y)

        return log_p_y

    def backpropagation(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        output = Variable(
            torch.from_numpy(output).long(),
            requires_grad=False
        )

        # Zero gradients
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.data.zero_()

        # Compute negative log-likelihood loss
        log_p_y = self._log_forward(input)
        cost = self.loss(log_p_y, output)
        # Use autograd to compute the backward pass.
        cost.backward()

        num_parameters = len(self.parameters)
        # Update parameters
        gradient_parameters = []
        for index in range(0, num_parameters):
            gradient_parameters.append(self.parameters[index].grad.data)

        return gradient_parameters
