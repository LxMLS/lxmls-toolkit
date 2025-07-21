from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import Counter, defaultdict
from math import log

dataset = load_dataset("emad12/stock_tweets_sentiment", split="train")
indices = list(range(len(dataset)))
shuffle(indices)
train_idx = indices[:-2000]
dev_idx = indices[-2000:-1000]
test_idx = indices[-1000:]

# output domain
sentiments = [-1, 0, 1]

smooth = 1

# example backward
# https://pytorch.org/docs/stable/notes/extending.html


def empirical_partial_derivative(function, dinput, indices=(0, 0, 0), h=1e-6):
    """
    Computes the approximate partial derivative of a FUNCTION by using a finite difference

    taking a numpy array or an scalar as input and
    INDICES = (output_index, input_index, batch/time_index)
    """
    # sanity checks
    assert x.ndim == 2, "expected x to have two dimensions"
    assert isinstance(x, np.ndarray), "expected x to be an np.array of data"
    assert isinstance(indices, tuple), "expected indices to be a tuple of indices"
    assert len(indices) == 3, "expected indices to have 3 dimensions"

    # numpy array at INDICES
    dinput_delta = dinput.copy()
    # perturb the input with a very small but finite constant
    dinput_delta[indices[0], indices[2]] += h
    # see the finite difference in the output
    return (function(dinput_delta)[indices[1], indices[2]] - function(dinput)[indices[1], indices[2]]) / h


def softmax(x):
    assert x.ndim == 2, "expected x to have two dimensions"
    assert isinstance(x, np.ndarray), "expected x to be a np.array"
    denominator = np.exp(x - x.max())
    return denominator / denominator.sum(0, keepdims=True)


def jacobian_softmax(x):
    assert x.ndim == 2, "expected x to have two dimensions"
    assert isinstance(x, np.ndarray), "expected x to be a np.array"
    # 3D Tensor where dimensions (1, 0) are a diagonal matrix of ones. Dim 2 is the batch/time dimension
    # shape = (N, N, T)
    diagonal = np.eye(x.shape[0])[:, :, None]
    # outer product through broadcasting
    return (diagonal - softmax(x)[:, None, :]) * softmax(x)[None, :, :]


def softmax_backward(e, x):
    jacobian = np.outer(softmax(x), (np.eye(x.shape[0]) - softmax(x)))
    return np.dot(jacobian, e)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def jacobian_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def test_jacobian():

    # -> (N, T)
    x = np.random.randn(10, 3)
    # (N, T) -> (N, T)
    p = softmax(x)
    # (N, T) -> (N, N, T)
    J_p = jacobian_softmax(x)
    # check
    index = (1, 0, 0)
    dp_dx = empirical_partial_derivative(softmax, x, index)
    assert np.allclose(dp_dx, J_p[index])


def mlp(x, weights, biases):
    activations = x
    for n in range(len(weights)):
        linear_activations = np.dot(weights[n], activations) + bias[n])
        if n + 1 < num_layers:
            activations = sigmoid(linear_activations)
        else:
            activations = softmax(linear_activations)
    return activations


def mlp_backprop(e, x, weights, biases):

    num_layers = len(weights)
    for n in range(num_layers)[::-1]:
        linear_activations = np.dot(weights[n], activations) + bias[n])
        if n + 1 < num_layers:
            activations = sigmoid(linear_activations)
        else:
            activations = softmax(linear_activations)
    return activations


def test_mlp():

    dimensions = [10, 20, 10]
    weights = [np.random.rand(dimensions[n+1], dimensions[n]) for n in range(len(dimensions)-1)]
    biases = [np.random.rand(dimensions[n+1], 1) for n in range(len(dimensions)-1)]

    x = np.random.randn(10, 3)

    p = mlp(x, weights, biases)


if __name__ == "__main__":
    # empirical_partial_derivative(sigmoid, np.array([[0.2, -0.4, 0.5], [0.1, 0.3, -0.8]]), (1, 1))
