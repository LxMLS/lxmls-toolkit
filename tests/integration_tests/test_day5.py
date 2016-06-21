from __future__ import division
from __future__ import print_function

import numpy as np
import pytest
import theano
import theano.tensor as T

import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd
import lxmls.readers.sentiment_reader as srs

tolerance = 1e-5
np.random.seed(4242)

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
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 2]
    actvfunc = ['softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc)

    hat_train_y = mlp.forward(train_x)
    hat_test_y = mlp.forward(test_x)

    acc_train = sgd.class_acc(hat_train_y, train_y)[0]
    acc_test = sgd.class_acc(hat_test_y, test_y)[0]
    assert abs(acc_train - 0.518750) < tolerance
    assert abs(acc_test - 0.542500) < tolerance


def test_exercise_2(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc)

    sgd.SGD_train(mlp, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.991875) < tolerance
    assert abs(acc_test - 0.79) < tolerance


def test_exercise_3(train_data, test_data):
    train_x, train_y = train_data
    test_x, test_y = test_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp = dl.NumpyMLP(geometry, actvfunc)

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
    mlp_a = dl.NumpyMLP(geometry, actvfunc)
    mlp_b = dl.TheanoMLP(geometry, actvfunc)

    # Check Numpy and Theano match
    resa = mlp_a.forward(test_x)[:, :10]
    resb = mlp_b.forward(test_x)[:, :10]
    assert np.allclose(resa, resb, rtol=tolerance)


def test_exercise_5(train_data):
    train_x, train_y = train_data
    I = train_x.shape[0]
    geometry = [I, 20, 2]
    actvfunc = ['sigmoid', 'softmax']
    mlp_a = dl.NumpyMLP(geometry, actvfunc)

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
    expected_gradients = np.array([[6.87651501e-04, 1.35650430e-04, 1.65650193e-04, 8.86584336e-04, 1.65086491e-03, 2.64808679e-04, 3.32879597e-03,
                                    2.61767047e-04, 3.41453047e-04, 7.21670337e-04],
                                   [-4.28321380e-04, -9.72922638e-05, -1.20215780e-04, -3.22285991e-04, -1.12260600e-03, -1.92079944e-04, -4.07377129e-03,
                                    -3.25119637e-04, -3.03224864e-04, -5.60586798e-04],
                                   [-8.31622967e-04, -2.21656061e-04, -2.27499288e-04, -7.64554006e-04, -2.16453545e-03, -5.10497437e-04, -5.49541344e-03,
                                    -7.76217333e-04, -5.77943456e-04, -1.16165572e-03],
                                   [1.58639047e-04, 2.96050183e-05, 4.10512014e-05, 1.97674910e-04, 3.48940143e-04, 8.73059516e-05, 6.78780131e-04,
                                    1.53785995e-04, 1.29834901e-04, 2.36391685e-04],
                                   [8.41569893e-04, 1.36813352e-04, 1.93333400e-04, 6.30826641e-04, 2.07568782e-03, 2.20454164e-04, 2.46769999e-03,
                                    6.68382428e-04, 3.15245620e-04, 1.07976781e-03],
                                   [-3.70163843e-05, -6.25172531e-06, -6.99608029e-06, -4.69461856e-05, -8.53419754e-05, 3.16602057e-06, -2.68644899e-04,
                                    -1.87799295e-05, -2.02376227e-05, -3.93226851e-05],
                                   [8.63995787e-04, 8.50400358e-05, 3.47095401e-04, 9.13967777e-04, 2.84152505e-03, 6.92705480e-04, 7.97452941e-03,
                                    4.48715106e-04, 5.90014623e-04, 1.34503064e-03],
                                   [2.71406468e-04, 4.01705299e-05, 7.80083288e-05, 2.05791387e-04, 5.95059906e-04, 1.57504709e-04, 2.77044809e-03,
                                    1.96606015e-04, 1.30796355e-04, 3.61806315e-04],
                                   [1.12119150e-03, 2.49485835e-04, 3.34197907e-04, 1.50915372e-03, 2.79914914e-03, 7.12211196e-04, 8.26003951e-03,
                                    9.75002352e-04, 8.55790142e-04, 1.68570945e-03],
                                   [4.95447895e-04, 1.33220618e-04, 1.40102555e-04, 7.15094880e-04, 1.24540411e-03, 2.57808463e-05, 6.47452089e-04,
                                    4.63321513e-04, 2.70073731e-04, 6.67939190e-04]])

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
    mlp_a = dl.NumpyMLP(geometry, actvfunc)
    sgd.SGD_train(mlp_a, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp_a.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp_a.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.991875) < tolerance
    assert abs(acc_test - 0.79) < tolerance

    # Theano grads
    mlp_b = dl.TheanoMLP(geometry, actvfunc)
    sgd.SGD_train(mlp_b, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
    acc_train = sgd.class_acc(mlp_b.forward(train_x), train_y)[0]
    acc_test = sgd.class_acc(mlp_b.forward(test_x), test_y)[0]
    assert abs(acc_train - 0.991875) < tolerance
    assert abs(acc_test - 0.79) < tolerance

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
