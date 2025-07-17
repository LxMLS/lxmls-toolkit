import numpy as np
import torch
from lxmls.deep_learning.mlp import MLP


def cast_float(variable_np):
    variable = torch.from_numpy(variable_np).float()
    variable.requires_grad = True
    return variable


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
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_function = torch.nn.NLLLoss()

    # TODO: Move these outside fo the class as in the numpy case
    def _log_forward(self, X):
        """
        Forward pass
        """

        # Ensure the type matches torch type
        X = cast_float(X)

        tilde_z = X

        # ----------
        # Solution to Exercise 6.4
        for n in range(self.num_layers - 1):

            # Get weights and bias of the layer (even and odd positions)
            weight, bias = self.parameters[n]

            # Linear transformation
            z = torch.matmul(tilde_z, torch.t(weight)) + bias

            # Non-linear transformation
            tilde_z = torch.sigmoid(z)

        # Get weights and bias of the layer (even and odd positions)
        weight, bias = self.parameters[self.num_layers - 1]

        # Linear transformation
        z = torch.matmul(tilde_z, torch.t(weight)) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = self.log_softmax(z)

        # End of solution to Exercise 6.4
        # ----------

        return log_tilde_z

    def gradients(self, X, y):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        true_class = torch.from_numpy(y).long()

        # Compute negative log-likelihood loss
        _log_forward = self._log_forward(X)
        loss = self.loss_function(_log_forward, true_class)
        # Use autograd to compute the backward pass.
        loss.backward()

        nabla_parameters = []
        for n in range(self.num_layers):
            weight, bias = self.parameters[n]
            nabla_parameters.append([weight.grad.data, bias.grad.data])
        return nabla_parameters

    def predict(self, X):
        """
        Predict model outputs given input
        """
        log_forward = self._log_forward(X).data.numpy()
        return np.argmax(log_forward, axis=1)

    def update(self, X, y):
        """
        Update model parameters given batch of data
        """
        gradients = self.gradients(X, y)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        for m in range(self.num_layers):
            # Update weight
            self.parameters[m][0].data -= learning_rate * gradients[m][0]
            # Update bias
            self.parameters[m][1].data -= learning_rate * gradients[m][1]

        # Zero gradients
        for n in range(self.num_layers):
            weight, bias = self.parameters[n]
            weight.grad.data.zero_()
            bias.grad.data.zero_()
