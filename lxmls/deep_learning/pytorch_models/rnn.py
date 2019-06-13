from __future__ import division

import numpy as np
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from lxmls.deep_learning.rnn import RNN
# To sample from model
from itertools import chain


def cast_float(variable, grad=True):
    return Variable(torch.from_numpy(variable).float(), requires_grad=grad)


def cast_int(variable, grad=True):
    return Variable(torch.from_numpy(variable).long(), requires_grad=grad)


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
        # Solution to Exercise 6.2

        # Word Embeddings
        z_e = self.embedding_layer(input)

        # Recurrent layer
        h = Variable(torch.FloatTensor(1, hidden_size).zero_())
        hidden_variables = []
        for t in range(nr_steps):

            # Linear
            z_t = torch.matmul(z_e[t, :], torch.t(W_x)) + \
                torch.matmul(h, torch.t(W_h))

            # Non-linear (sigmoid)
            h = torch.sigmoid(z_t)

            hidden_variables.append(h)

        # Output layer
        h_out = torch.cat(hidden_variables, 0)
        y = torch.matmul(h_out, torch.t(W_y))

        # Log-Softmax
        log_p_y = self.logsoftmax(y)

        # End of solution to Exercise 6.2
        # ----------

        return log_p_y

    def backpropagation(self, input, output):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """

        # Ensure the type matches torch type
        output = Variable(torch.from_numpy(output).long())

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
    Basic RNN with forward-pass and gradient computation in Pytorch. Uses
    native Pytorch RNN
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
        # TODO: Switch here to RL loss depending on config
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
        input = cast_int(input)

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
        output = cast_int(output, grad=False)

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


class PolicyRNN(FastPytorchRNN):
    """
    Basic RNN with forward-pass and gradient computation in Pytorch. Uses
    native Pytorch RNN
    """

    def __init__(self, **config):

        # This will initialize
        # self.num_layers
        # self.config
        # self.parameters
        FastPytorchRNN.__init__(self, **config)
        if config.get('RL', False):
            self.loss = self.reinforce_loss
        else:
            self.loss = torch.nn.NLLLoss()
        self._gamma = config.get('gamma', 0.9)
        self._maxL = config.get('maxL', None)

    def _sample(self, input=None):
        """
        Return one sample from the model and its minus log-probability
        sample from current policy
        :return the samples and its neg. log probabilities
        """
        logits = self._log_forward(input)
        distribution = Categorical(
            logits=logits.view(-1, logits.size(-1))
        )
        samples = distribution.sample()
        log_probs = -distribution.log_prob(samples)

        return samples, log_probs

    def torch_ind2onehot(self, tensor_shape, idx, dim):
        onehot = torch.FloatTensor(tensor_shape)
        onehot.zero_()
        onehot.scatter_(dim, idx, 1)
        return onehot

    def torch_batch2onehot(self, batch, depth):
        emb = torch.nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)

        return emb(batch)

    def VRreinforce_loss(self, log_p_y, loutput):
        """
        Computes the REINFORCE loss over a batch of sequences
        :param log_p_y: (Batch_size*Len)x dim
        :param out: packed sequence w data (Batch_size*Len) and batch_sizes
        :return: loss as a real value
        """

        out = self.pack(loutput)
        output = out.data
        out_vec = self.torch_ind2onehot(
            log_p_y.shape,
            output.reshape(-1, 1), -1
        )
        # torch.sum
        cost = (torch.exp(log_p_y) - out_vec)**2
        # cost to go always positive
        R = self.cost_to_go(
            cost,
            out.batch_sizes,
            gamma=self._gamma,
            dim=0
        )
        # compute baseline
        b = self.baseline(R)
        # Calculate loss
        selected_logprobs = - (R - b) * log_p_y
        # avg Sum_t cost to go over sequences
        loss = selected_logprobs.sum() / float(len(out.batch_sizes))
        return loss

    def reinforce_loss(self, log_p_y, loutput):
        """
        Computes the REINFORCE loss over a batch of sequences
        :param log_p_y: (Batch_size*Len)x dim
        :param out: packed sequence w data (Batch_size*Len) and batch_sizes
        :return: loss as a real value
        """
        sizes = [len(i) for i in loutput]
        out = self.pack(loutput)
        output = out.data
        pred = torch.max(log_p_y, dim=1)[1]
        # * lengths
        cost = (pred == output).float()
        R = self.cost_to_go(
            cost.reshape(-1, 1),
            sizes,
            gamma=self._gamma,
            dim=0
        )
        # Calculate loss
        selected_logprobs = -R.reshape(-1) * \
            log_p_y[np.arange(len(output)), output]
        # sum in time and class dimension mean over batch size
        loss = selected_logprobs.sum() / float(len(out.batch_sizes))
        return loss

    def cost_to_go(self, rwd, sizes=None, gamma=0.99, dim=1):
        # calculate cumulative cost to go
        if sizes is None:
            col = [gamma**i for i in range(rwd.shape[dim])]
            row = np.zeros((rwd.shape[dim]), dtype=float)
            row[0] = col[0]
            gammas = scipy.linalg.toeplitz(row, col)
            gammas = cast_float(gammas, grad=False)
            R = torch.matmul(gammas, rwd)
        else:
            j = 0
            ctg = []
            for i, T in enumerate(sizes):
                col = [gamma ** i for i in range(T)]
                row = np.zeros((T), dtype=float)
                row[0] = col[0]
                gammas = scipy.linalg.toeplitz(row, col)
                gammas = cast_float(gammas, grad=False)
                rt = torch.matmul(gammas, rwd[j:j+T, :])
                j += T
                ctg.append(rt)
            R = torch.cat(ctg, dim=0)
        return R

    def _log_forward(self, linput):
        """
        Forward pass
        """
        # Ensure the type matches torch type
        input = self.pack(linput)
        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        embedding_size, vocabulary_size = W_e.shape

        # FORWARD PASS COMPUTATION GRAPH

        # Word Embeddings

        z_e = self.embedding_layer(input.data)
        pack_z_e = torch.nn.utils.rnn.pack_sequence(
            self.unpack(z_e, input.batch_sizes)
        )

        # RNN
        self.h, _ = self.rnn(pack_z_e)

        # Output layer
        y = torch.matmul(self.h.data, torch.t(W_y))

        # Log-Softmax
        log_p_y = self.logsoftmax(y)

        return log_p_y

    def backpropagation(self, linput, loutput):
        """
        Computes the gradients of the network with respect to cross entropy
        error cost
        """
        # Zero gradients
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.data.zero_()

        # Compute negative log-likelihood loss

        log_p_y = self._log_forward(linput)
        cost = self.loss(log_p_y, loutput)
        # Use autograd to compute the backward pass.
        cost.backward()

        num_parameters = len(self.parameters)
        # Update parameters
        gradient_parameters = []
        for index in range(0, num_parameters):
            gradient_parameters.append(self.parameters[index].grad.data)

        return gradient_parameters

    @staticmethod
    def batch_var_lenlist(loutput):
        lengths = [(len(i), j) for j, i in enumerate(loutput)]
        lengths = sorted(lengths, reverse=True)
        out_var = [
            cast_int(loutput[i], grad=False) for i in list(zip(*lengths))[1]
        ]
        pad_output = torch.nn.utils.rnn.pad_sequence(out_var)
        output = torch.nn.utils.rnn.pack_padded_sequence(
            pad_output,
            list(zip(*lengths))[0], batch_first=False
        )
        return output

    def pack(self, loutput, grad=False):
        lengths = [(len(i), j) for j, i in enumerate(loutput)]
        lengths = sorted(lengths, key=lambda x: x[0], reverse=True)
        padded_output = np.zeros((self._maxL, len(loutput)), dtype=float)
        for i in range(len(loutput)):
            padded_output[:len(loutput[i]), i] = np.asarray(loutput[i])
        out_var = cast_int(padded_output, grad=grad)
        output = torch.nn.utils.rnn.pack_padded_sequence(
            out_var,
            list(list(zip(*lengths))[0]), batch_first=False
        )
        return output

    @staticmethod
    def unpack(mat, batch_sizes):
        return [
            mat[sum(batch_sizes[:i]):sum(batch_sizes[:i + 1]), :]
            for i in range(len(batch_sizes))
        ]

    def predict_loss(self, linput, loutput):
        prediction = self.predict(linput)
        output = self.pack(loutput).data.numpy()
        return prediction == output

    def baseline(self, R):
        """
        compute baseline as E(R| w_1:t-1, a_1:t-a) = < h_t, w >
        :return: <h_t,w>
        """
        # estimate baseline weights with all batch samples using OLS
        H = self.h.data
        H = H.detach().numpy()
        w = np.dot(np.linalg.pinv(H), R.data.detach().numpy())
        b = np.dot(H, w)
        return cast_float(b, grad=False)
