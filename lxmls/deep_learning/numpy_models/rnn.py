import numpy as np
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, X):
        """
        Predict model outputs given input
        """
        log_p_y = self.log_forward(X)[0]
        return np.argmax(log_p_y, axis=1)

    def update(self, X, y):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(X, y)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m] -= learning_rate * gradients[m]

    def log_forward(self, X):

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        hidden_size = W_h.shape[0]
        nr_steps = X.shape[0]

        # Embedding layer
        z_e = W_e[X, :]

        # Recurrent layer
        h = np.zeros((nr_steps + 1, hidden_size))
        for t in range(nr_steps):

            # Linear
            z_t = W_x.dot(z_e[t, :]) + W_h.dot(h[t, :])

            # Non-linear
            h[t + 1, :] = 1.0 / (1 + np.exp(-z_t))

        # Output layer
        y = h[1:, :].dot(W_y.T)

        # Softmax
        log_p_y = y - logsumexp(y, axis=1, keepdims=True)

        return log_p_y, y, h, z_e, X  # why does this return its own input?

    def backpropagation(self, X, y):

        '''
        Compute gradientes, with the back-propagation method
        inputs:
            X: matrix with the (embedding) indicies of the words of a
                sentence
            y: vector with the indicies of the tags for each word of
                        the sentence outputs:
            gradient_parameters: vector with parameters gradientes
        '''

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        nr_steps = X.shape[0]

        log_p_y, y, h, z_e, x = self.log_forward(X)
        p_y = np.exp(log_p_y)

        # Initialize gradients with zero entrances
        gradient_W_e = np.zeros(W_e.shape)
        gradient_W_x = np.zeros(W_x.shape)
        gradient_W_h = np.zeros(W_h.shape)
        gradient_W_y = np.zeros(W_y.shape)

        # ----------
        # Solution to Exercise 6.1

        # Gradient of the cost with respect to the last linear model
        I = index2onehot(y, W_y.shape[0])
        error = - (I - p_y) / nr_steps

        # backward pass, with gradient computation
        error_h_next = np.zeros_like(h[0, :])
        for t in reversed(range(nr_steps)):

            # Output linear
            error_h = np.dot(W_y.T, error[t, :]) + error_h_next

            # Non-linear
            error_raw = h[t+1, :] * (1. - h[t+1, :]) * error_h

            # Hidden-linear
            error_h_next = np.dot(W_h.T, error_raw)

            # Weight gradients
            gradient_W_y += np.outer(error[t, :], h[t+1, :])
            gradient_W_h += np.outer(error_raw, h[t, :])
            gradient_W_x += np.outer(error_raw, z_e[t, :])
            gradient_W_e[x[t], :] += W_x.T.dot(error_raw)

        # End of Solution to Exercise 6.1
        # ----------

        # Normalize over sentence length
        gradient_parameters = [
            gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
        ]

        return gradient_parameters

    def cross_entropy_loss(self, X, y):
        """Cross entropy loss"""
        nr_steps = X.shape[0]
        log_probability = self.log_forward(X)[0]
        return -log_probability[range(nr_steps), y].mean()
