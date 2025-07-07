import numpy as np
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, model_input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self.log_forward(model_input)[0])
        return np.argmax(p_y, axis=1)

    def update(self, model_input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(model_input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m] -= learning_rate * gradients[m]

    def log_forward(self, model_input):

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        hidden_size = W_h.shape[0]
        nr_steps = model_input.shape[0]
        nr_tokens = W_e.shape[1]

        # Embedding layer
        input_ohe = index2onehot(model_input, nr_tokens)
        z_e = input_ohe @ W_e.T

        # Recurrent layer
        h = np.zeros((nr_steps + 1, hidden_size))
        for t in range(nr_steps):

            # Linear
            z_t = W_x.dot(z_e[t, :]) + W_h.dot(h[t, :])

            # Non-linear
            h[t+1, :] = 1.0 / (1 + np.exp(-z_t))

        # Output layer
        y = h[1:, :].dot(W_y.T)

        # Softmax
        log_p_y = y - logsumexp(y, axis=1, keepdims=True)

        return log_p_y, y, h, z_e, model_input

    def backpropagation(self, model_input, output) -> list[np.ndarray]:
        """
        Compute gradients for the RNN, with the back-propagation method.

        Inputs:
            x: vector with the (embedding) indices of the words of a
                sentence
            outputs: vector with the indices of the tags for each word of
                        the sentence
        Outputs:
            gradient_parameters (list[np.ndarray]): List with W_e, W_x, W_h, W_y parameters' gradients
        """
        # print(f"Model input shape: {model_input.shape}, Output shape: {output.shape}")

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        nr_steps = input.shape[0]

        log_p_y, y, h, z_e, x = self.log_forward(input)
        p_y = np.exp(log_p_y)

        # Initialize gradients with zero entrances
        gradient_W_e = np.zeros(W_e.shape)
        gradient_W_x = np.zeros(W_x.shape)
        gradient_W_h = np.zeros(W_h.shape)
        gradient_W_y = np.zeros(W_y.shape)

        # ----------
        # Solution to Exercise 1

        raise NotImplementedError("Implement Exercise 1")

        # End of Solution to Exercise 1
        # ----------

        # Normalize over sentence length
        gradient_parameters = [
            gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
        ]

        return gradient_parameters

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        nr_steps = input.shape[0]
        log_probability = self.log_forward(input)[0]
        return -log_probability[range(nr_steps), output].mean()
