import torch


class PytorchLogLinear(Model):

    def __init__(self, **config):

        # Initialize parameters
        weight_shape = (config['input_size'], config['num_classes'])
        # after Xavier Glorot et al
        weight_np = glorot_weight_init(weight_shape, 'softmax')
        self.learning_rate = config['learning_rate']

        # IMPORTANT: Cast to pytorch format
        self.weight = torch.from_numpy(weight_np).float()
        self.weight.requires_grad = True

        self.bias = torch.zeros(1, config['num_classes'], requires_grad=True)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_function = torch.nn.NLLLoss()

    def _log_forward(self, input=None):
        """Forward pass of the computation graph in logarithm domain (pytorch)"""

        # IMPORTANT: Cast to pytorch format
        input = torch.from_numpy(input).float()

        # Linear transformation
        z =  torch.matmul(input, torch.t(self.weight)) + self.bias

        # Softmax implemented in log domain
        log_tilde_z = self.log_softmax(z)

        # NOTE that this is a pytorch class!
        return log_tilde_z

    def predict(self, input=None):
        """Most probable class index"""
        log_forward = self._log_forward(input).data.numpy()
        return np.argmax(log_forward, axis=1)

    def update(self, input=None, output=None):
        """Stochastic Gradient Descent update"""

        # IMPORTANT: Class indices need to be casted to LONG
        true_class = torch.from_numpy(output).long()

        # Compute negative log-likelihood loss
        loss = self.loss_function(self._log_forward(input), true_class)

        # Use autograd to compute the backward pass.
        loss.backward()

        # SGD update
        self.weight.data -= self.learning_rate * self.weight.grad.data
        self.bias.data -= self.learning_rate * self.bias.grad.data

        # Zero gradients
        self.weight.grad.data.zero_()
        self.bias.grad.data.zero_()

        return loss.data.numpy()
