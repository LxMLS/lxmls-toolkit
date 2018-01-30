from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
from lxmls.deep_learning.mlp import MLP


def cast_float(variable):
    return Variable(torch.from_numpy(variable).float(), requires_grad=True)


class PytorchMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Pytorch
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

        # Need to cast all weights
        for n in range(self.num_layers):
            # Get weigths and bias of the layer (even and odd positions)
            weight, bias = self.parameters[n]
            self.parameters[n] = [cast_float(weight), cast_float(bias)]

        # Initialize some functions that we will need
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    # TODO: Move these outside fo the class as in the numpy case
    def _log_forward(self, input):
        """
        Forward pass
        """

        # Ensure the type matches torch type
        input = cast_float(input)

        # Input
        tilde_z = input

        # ----------
        # Solution to Exercise 6.4
        for n in range(self.num_layers - 1):

            # Get weigths and bias of the layer (even and odd positions)
            weight, bias = self.parameters[n]

            # Linear transformation
            z = torch.matmul(tilde_z, torch.t(weight)) + bias

            # Non-linear transformation
            tilde_z = torch.sigmoid(z)

        # Get weigths and bias of the layer (even and odd positions)
        weight, bias = self.parameters[self.num_layers - 1]

        # Linear transformation
        z = torch.matmul(tilde_z, torch.t(weight)) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = self.logsoftmax(z)

        # End of solution to Exercise 6.4
        # ----------

        return log_tilde_z

    def gradients(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        true_class = Variable(
            torch.from_numpy(output).long(),
            requires_grad=False
        )

        # Compute negative log-likelihood loss
        _log_forward = self._log_forward(input)
        loss = torch.nn.NLLLoss()(_log_forward, true_class)
        # Use autograd to compute the backward pass.
        loss.backward()

        nabla_parameters = []
        for n in range(self.num_layers):
            weight, bias = self.parameters[n]
            nabla_parameters.append([weight.grad.data, bias.grad.data])
        return nabla_parameters

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_forward = self._log_forward(input).data.numpy()
        return np.argmax(np.exp(log_forward), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.gradients(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        for m in np.arange(self.num_layers):
            # Update weight
            self.parameters[m][0].data -= learning_rate * gradients[m][0]
            # Update bias
            self.parameters[m][1].data -= learning_rate * gradients[m][1]

        # Zero gradients
        for n in np.arange(self.num_layers):
            weight, bias = self.parameters[n]
            weight.grad.data.zero_()
            bias.grad.data.zero_()
