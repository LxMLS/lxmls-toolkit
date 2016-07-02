from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import theano
import theano.tensor as T
from numpy.random import RandomState

import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd
import lxmls.readers.sentiment_reader as srs

tolerance = 1e-5
seed = 4242

# Model parameters
n_iter = 3
bsize = 5
lrate = 0.05


@pytest.fixture(scope='module')
def sentiment_corpus():
    return srs.SentimentCorpus("books")


@pytest.fixture(scope='module')
def train_data(sentiment_corpus):
    return sentiment_corpus.train_X.T, sentiment_corpus.train_y[:, 0]


@pytest.fixture(scope='module')
def test_data(sentiment_corpus):
    return sentiment_corpus.test_X.T, sentiment_corpus.test_y[:, 0]


def test_exercise_1(train_data, test_data):
    np.random.seed(seed)
    print('rnd: {}'.format(np.random.get_state()[1][:4]))

    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 2]
    actvfunc = ['softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))

    hat_train_y = mlp.forward(train_x)
    hat_test_y = mlp.forward(test_x)

    acc_train = sgd.class_acc(hat_train_y, train_y)[0]
    acc_test = sgd.class_acc(hat_test_y, test_y)[0]
    assert abs(acc_train - 0.50375) < tolerance
    assert abs(acc_test - 0.4475) < tolerance


def test_exercise_2(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))

    sgd.SGD_train(mlp, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.99125) < tolerance
    assert abs(acc_test - 0.84) < tolerance


def test_exercise_3(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))

    x = test_x
    W1, b1 = mlp.params[0:2]
    z1 = np.dot(W1, x) + b1
    tilde_z1 = 1. / (1+np.exp(-z1))
    assert isinstance(tilde_z1, np.ndarray)

    _x = T.matrix('x')
    _W1 = theano.shared(value=W1, name='W1', borrow=True)
    _b1 = theano.shared(value=b1, name='b1', borrow=True, broadcastable=(False, True))

    # Perceptron
    _z1 = T.dot(_W1, _x) + _b1
    _tilde_z1 = T.nnet.sigmoid(_z1)

    layer1 = theano.function([_x], _tilde_z1)

    assert np.allclose(tilde_z1, layer1(x.astype(theano.config.floatX)), rtol=tolerance)


def test_exercise_4(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp_a = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))
    mlp_b = dl.TheanoMLP(geometry, actvfunc, rng=RandomState(seed))

    # Check Numpy and Theano match
    resa = mlp_a.forward(test_x)[:, :10]
    resb = mlp_b.forward(test_x)[:, :10]
    assert np.allclose(resa, resb, rtol=tolerance)


def test_exercise_5(train_data):
    train_x, train_y = train_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp_a = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))

    W1, b1 = mlp_a.params[0:2]

    _x = T.matrix('x')
    _W1 = theano.shared(value=W1, name='W1', borrow=True)
    _b1 = theano.shared(value=b1, name='b1', borrow=True, broadcastable=(False, True))

    _z1 = T.dot(_W1, _x) + _b1
    _tilde_z1 = T.nnet.sigmoid(_z1)

    W2, b2 = mlp_a.params[2:4]

    _W2 = theano.shared(value=W2, name='W2', borrow=True)
    _b2 = theano.shared(value=b2, name='b2', borrow=True, broadcastable=(False, True))
    _z2 = T.dot(_W2, _tilde_z1) + _b2
    _tilde_z2 = T.nnet.softmax(_z2.T).T

    _y = T.ivector('y')

    _F = -T.mean(T.log(_tilde_z2[_y, T.arange(_y.shape[0])]))

    _nabla_F = T.grad(_F, _W1)
    nabla_F = theano.function([_x, _y], _nabla_F)

    x = train_x.astype(theano.config.floatX)
    y = train_y.astype('int32')
    gradients = nabla_F(x, y)[:10, :10]
    expected_gradients = np.array([[1.17097689e-04, 2.29797958e-04, 1.31026038e-04,
                                    -1.23928600e-04, 1.54682859e-03, 2.10400922e-04,
                                    -3.07062981e-05, 2.90462523e-04, 3.86890011e-05,
                                    8.65776516e-05],
                                   [-1.07199294e-04, -4.36321055e-04, -2.06119969e-04,
                                    1.07280751e-04, -2.96771855e-03, -4.05524208e-04,
                                    1.08434017e-04, -6.03050125e-04, -8.66574939e-05,
                                    -1.02925071e-04],
                                   [7.37837175e-05, 1.13226096e-04, 8.66791423e-05,
                                    -1.10596322e-04, 1.07282954e-03, 1.26845815e-04,
                                    -6.33698425e-05, 1.81804188e-04, 2.04807998e-05,
                                    3.19311417e-05],
                                   [1.65634428e-04, 2.65380938e-04, 1.68474872e-04,
                                    -8.27166721e-05, 2.07753561e-03, 2.45483387e-04,
                                    -3.11423106e-04, 3.70193266e-04, 5.65979008e-05,
                                    1.20197891e-04],
                                   [-1.57517103e-04, -4.98654979e-04, -2.88112878e-04,
                                    1.09285571e-04, -2.82454826e-03, -4.64399711e-04,
                                    1.10226756e-03, -4.71803338e-04, -9.21129775e-05,
                                    -2.16984594e-04],
                                   [-2.72370419e-05, -7.48637956e-05, -5.19809330e-05,
                                    6.44237190e-05, -5.60225936e-04, -6.88599072e-05,
                                    -1.02237198e-04, -9.00221133e-05, -1.54118567e-05,
                                    5.63113291e-07],
                                   [5.49185157e-04, 8.06704582e-04, 6.69431007e-04,
                                    -6.46599659e-04, 7.69724433e-03, 6.85097974e-04,
                                    5.39098818e-04, 1.26229423e-03, 1.46950770e-04,
                                    1.48799912e-04],
                                   [8.77227396e-05, 2.73479470e-04, 1.40993078e-04,
                                    -9.29150251e-05, 1.88333750e-03, 2.42434737e-04,
                                    3.23509163e-04, 3.39385958e-04, 2.98555843e-05,
                                    1.01266554e-04],
                                   [3.02338828e-06, 1.50176963e-03, 8.82548115e-04,
                                    -3.41911021e-04, 1.04542207e-02, 9.10085620e-04,
                                    -4.13317114e-04, 1.81006489e-03, 2.46318330e-04,
                                    2.15817395e-04],
                                   [-2.74792935e-05, -6.35337842e-04, -4.32977451e-04,
                                    2.72895865e-04, -4.90612435e-03, -5.70313422e-04,
                                    1.54647017e-03, -7.89472747e-04, -1.52254402e-04,
                                    -2.48578495e-04]])

    assert np.allclose(gradients, expected_gradients, rtol=tolerance)


def test_exercise_6(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]

    train_x = train_x.astype(theano.config.floatX)
    train_y = train_y.astype('int32')

    _train_x = theano.shared(train_x, 'train_x', borrow=True)
    _train_y = theano.shared(train_y, 'train_y', borrow=True)

    _i = T.lscalar()
    get_tr_batch_y = theano.function([_i], _train_y[_i * bsize:(_i + 1) * bsize])

    i = 3
    resa = train_y[i * bsize:(i + 1) * bsize]
    resb = get_tr_batch_y(i)
    assert np.allclose(resa, resb, rtol=tolerance)

    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']

    # Numpy
    mlp_a = dl.NumpyMLP(geometry, actvfunc, rng=RandomState(seed))
    sgd.SGD_train(mlp_a, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp_a.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp_a.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.99125) < tolerance
    assert abs(acc_test - 0.84) < tolerance

    # Theano grads
    mlp_b = dl.TheanoMLP(geometry, actvfunc, rng=RandomState(seed))
    sgd.SGD_train(mlp_b, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp_b.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp_b.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.99125) < tolerance
    assert abs(acc_test - 0.84) < tolerance

    mlp_c = dl.TheanoMLP(geometry, actvfunc)
    _x = T.matrix('x')
    _y = T.ivector('y')
    _F = mlp_c._cost(_x, _y)
    updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_c.params]

    _j = T.lscalar()
    givens = {_x: _train_x[:, _j*bsize:(_j+1)*bsize],
              _y: _train_y[_j*bsize:(_j+1)*bsize]}

    batch_up = theano.function([_j], _F, updates=updates, givens=givens)
    n_batch = train_x.shape[1]/bsize + 1

    sgd.SGD_train(mlp_c, n_iter, batch_up=batch_up, n_batch=n_batch)
    acc_train = sgd.class_acc(mlp_c.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp_c.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.991875) < tolerance
    assert abs(acc_test - 0.79) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
