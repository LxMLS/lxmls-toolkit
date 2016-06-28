#!/usr/bin/python
"""
Deep learning day exercises corresponding to Chapter 5 of the LxMLS guide.
They must be run from ./lxmls-toolkit/ folder as

./lxmls/labs/day5.py

These are the solutions of the exercises. Students of the LxMLS school should
work with the student branch and ignore this.

In any case, to inspect this code you may want to use some debugging tools.
Have look at day0 of the lxmls_guide for some references. The simplest approach
is the pdb module included by default in Python. To insert a break point write
i your code

import pdb
ipdb.set_trace()

In Table 1 of day 0 of the guide you can find some useful commands to browse
through the code while in debug mode.

An alternative to pdb is ipdb. This works exactly the same but has colors and
autocomplete. You can install it with pip

sudo pip install ipdb

There is also a little modification you can do to get ipdb show more lines of
context in debug mode, see

http://stackoverflow.com/questions/6240887/how-can-i-make-ipdb-show-more-lines-of-context-while-debugging

For a more matlab-like environment you can use Spider or ipython. If you port
 this to ipython notebook let me know! (ramon@astudillo.com)
"""

import sys

sys.path.append('.')
import time

from ipdb import set_trace

print "\n######################",
print "\n   Exercise 5.1"
print "######################"

# TODO: Find another corpus (SemEval?) and cleaner parameter extraction
import numpy as np
import lxmls.readers.sentiment_reader as srs

scr = srs.SentimentCorpus("books")
train_x = scr.train_X.T
train_y = scr.train_y[:, 0]
test_x = scr.test_X.T
test_y = scr.test_y[:, 0]

# Define MLP (log linear)
import lxmls.deep_learning.mlp as dl

I = train_x.shape[0]
# Model parameters
geometry = [I, 2]
actvfunc = ['softmax']
# Instantiate model
mlp = dl.NumpyMLP(geometry, actvfunc)

# Play with the untrained MLP forward
hat_train_y = mlp.forward(train_x)
hat_test_y = mlp.forward(test_x)
# Compute accuracy
import lxmls.deep_learning.sgd as sgd

acc_train = sgd.class_acc(hat_train_y, train_y)[0]
acc_test = sgd.class_acc(hat_test_y, test_y)[0]
print "Untrained Log-linear Accuracy train: %f test: %f" % (acc_train, acc_test)

print "\n######################",
print "\n   Exercise 5.2"
print "######################\n"

# Define MLP
# Model parameters
geometry = [I, 20, 2]
actvfunc = ['sigmoid', 'softmax']
# Instantiate model
mlp = dl.NumpyMLP(geometry, actvfunc)

# Model parameters
n_iter = 5
bsize = 5
lrate = 0.05
# Train
sgd.SGD_train(mlp, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
acc_train = sgd.class_acc(mlp.forward(train_x), train_y)[0]
acc_test = sgd.class_acc(mlp.forward(test_x), test_y)[0]
print "MLP %s Model Amazon Sentiment Accuracy train: %f test: %f" % (geometry, acc_train, acc_test)

print "\n######################",
print "\n   Exercise 5.3"
print "######################"

# Numpy code
import numpy as np

x = test_x  # Test set
W1, b1 = mlp.params[0:2]  # Weigths and bias of fist layer
z1 = np.dot(W1, x) + b1  # Linear transformation
tilde_z1 = 1 / (1+np.exp(-z1))  # Non-linear transformation

# Theano code.
# NOTE: We use underscore to denote symbolic equivalents to Numpy variables.
# This is no Python convention!.
import theano
import theano.tensor as T

_x = T.matrix('x')
_W1 = theano.shared(value=W1, name='W1', borrow=True)
_b1 = theano.shared(value=b1, name='b1', borrow=True,
                    broadcastable=(False, True))

# Perceptron
_z1 = T.dot(_W1, _x) + _b1
_tilde_z1 = T.nnet.sigmoid(_z1)
# Keep in mind that naming variables is useful when debugging
_z1.name = 'z1'
_tilde_z1.name = 'tilde_z1'

# Compile
layer1 = theano.function([_x], _tilde_z1)

# Show computation graph
print "\nThis is my symbolic perceptron\n"
theano.printing.debugprint(_tilde_z1)

# Check Numpy and Theano mactch
if np.allclose(tilde_z1, layer1(x.astype(theano.config.floatX))):
    print "\nNumpy and Theano Perceptrons are equivalent"
else:
    set_trace()
    # raise ValueError, "Numpy and Theano Perceptrons are different"

print "\n######################",
print "\n   Exercise 5.4"
print "######################"
mlp_a = dl.NumpyMLP(geometry, actvfunc)
mlp_b = dl.TheanoMLP(geometry, actvfunc)

# To debug layer by layer, you may use
#
#     fwd1          = mlp_a.forward(test_x[:, :10], allOuts=True)
#
#     mlp_b_forward = theano.function([_x], mlp_b._forward(_x, allOuts=True))
#     fwd2          = mlp_b_forward(test_x[:, :10])
#
# Or e.g. to see the graph up to the second variable
#
#     theano.printing.debugprint(mlp_b._forward([_x], allOuts=True)[1])
#
# Be sure to check also the parameters of the model e.g.
#
#     mlp_a.params[0]
#
#     mlp_b.params[0].get_value()

# Show computation graph
print "\nThis is my symbolic forward\n"
theano.printing.debugprint(mlp_b._forward(T.matrix('x')))

# Check Numpy and Theano match
resa = mlp_a.forward(test_x)[:, :10]
resb = mlp_b.forward(test_x)[:, :10]
if np.allclose(resa, resb):
    print "\nNumpy and Theano Forward pass are equivalent"
else:
    set_trace()
    # raise ValueError, "Numpy and Theano Forward are different"

# FOR DEBUGGING PURPOSES
# Check Numpy and Theano match
# resas = mlp_a.grads(test_x[:, :10], test_y[:10])
# resbs = mlp_b.grads(test_x[:, :10], test_y[:10])
# if np.all([np.allclose(ra, rb) for ra, rb in zip(resas, resbs)]):
#    print "DEBUG: Numpy and Theano Gradients pass are equivalent"
# else:
#    set_trace()
#    #raise ValueError, "\nDEBUG: Numpy and Theano Gradients are different"

print "\n######################",
print "\n   Exercise 5.5"
print "######################"

W2, b2 = mlp_a.params[2:4]

# Second layer symbolic variables
_W2 = theano.shared(value=W2, name='W2', borrow=True)
_b2 = theano.shared(value=b2, name='b2', borrow=True,
                    broadcastable=(False, True))
_z2 = T.dot(_W2, _tilde_z1) + _b2
_tilde_z2 = T.nnet.softmax(_z2.T).T

# Ground truth
_y = T.ivector('y')

# Cost
_F = -T.mean(T.log(_tilde_z2[_y, T.arange(_y.shape[0])]))

# Gradient
_nabla_F = T.grad(_F, _W1)
nabla_F = theano.function([_x, _y], _nabla_F)

# Print computation graph
print "\nThis is my softmax classification cost\n"
theano.printing.debugprint(_F)

# FOR DEBUGGING PURPOSES
# print "\nThis is my classification cost weight gradient\n"
# theano.printing.debugprint(nabla_F)

print "\n######################",
print "\n   Exercise 5.6"
print "######################"

# Understanding the mini-batch function and givens/updates parameters

# Cast data into the types and shapes used in the theano graph
# IMPORTANT: This is the main source of errors when beginning with theano
train_x = train_x.astype(theano.config.floatX)
train_y = train_y.astype('int32')

# Store data as shared variables
# NOTE: This will push the data into the GPU memory when used
_train_x = theano.shared(train_x, 'train_x', borrow=True)
_train_y = theano.shared(train_y, 'train_y', borrow=True)

# Create a symbolic variable returning a batch of samples
_i = T.lscalar()
get_tr_batch_y = theano.function([_i], _train_y[_i*bsize:(_i+1)*bsize])

# Check Numpy and Theano match
i = 3
resa = train_y[i*bsize:(i+1)*bsize]
resb = get_tr_batch_y(i)
if np.allclose(resa, resb):
    print "\nNumpy and Theano  Mini-Batch pass are equivalent\n"
else:
    set_trace()
    # raise ValueError, "Numpy and Theano Mini-Batch are different"

# Compare Numpy, Theano and Theano compiled

# Numpy
geometry = [I, 20, 2]
actvfunc = ['sigmoid', 'softmax']
mlp_a = dl.NumpyMLP(geometry, actvfunc)
#
init_t = time.clock()
sgd.SGD_train(mlp_a, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
print "\nNumpy version took %2.2f sec" % (time.clock() - init_t)
acc_train = sgd.class_acc(mlp_a.forward(train_x), train_y)[0]
acc_test = sgd.class_acc(mlp_a.forward(test_x), test_y)[0]
print "Amazon Sentiment Accuracy train: %f test: %f\n" % (acc_train, acc_test)

# Theano grads
mlp_b = dl.TheanoMLP(geometry, actvfunc)
init_t = time.clock()
sgd.SGD_train(mlp_b, n_iter, bsize=bsize, lrate=lrate, train_set=(train_x, train_y))
print "\nCompiled gradient version took %2.2f sec" % (time.clock() - init_t)
acc_train = sgd.class_acc(mlp_b.forward(train_x), train_y)[0]
acc_test = sgd.class_acc(mlp_b.forward(test_x), test_y)[0]
print "Amazon Sentiment Accuracy train: %f test: %f\n" % (acc_train, acc_test)

# Theano compiled batch

# Cast data into the types and shapes used in the theano graph
# IMPORTANT: This is the main source of errors when beginning with theano
train_x = train_x.astype(theano.config.floatX)
train_y = train_y.astype('int32')
# Store data as shared variables
# NOTE: This will push the data into the GPU memory when used
_train_x = theano.shared(train_x, 'train_x', borrow=True)
_train_y = theano.shared(train_y, 'train_y', borrow=True)

# Model
mlp_c = dl.TheanoMLP(geometry, actvfunc)

# Define givens variables to be used in the batch update
# Get symbolic variables returning a mini-batch of data

# Define updates variable. This is a list of gradient descent updates
# The output is a list following theano.function updates parameter. This
# consists on a list of tuples with each parameter and update rule
_x = T.matrix('x')
_y = T.ivector('y')
_F = mlp_c._cost(_x, _y)
updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_c.params]

# Givens maps input and target to a mini-batch of inputs and targets
_j = T.lscalar()
givens = {_x: _train_x[:, _j*bsize:(_j+1)*bsize],
          _y: _train_y[_j*bsize:(_j+1)*bsize]}

# Define the batch update function. This will return the cost of each batch
# and update the MLP parameters at the same time using updates
batch_up = theano.function([_j], _F, updates=updates, givens=givens)
n_batch = train_x.shape[1]/bsize + 1

init_t = time.clock()
sgd.SGD_train(mlp_c, n_iter, batch_up=batch_up, n_batch=n_batch)
print "\nTheano compiled batch update version took %2.2f sec" % (time.clock() - init_t)
init_t = time.clock()

acc_train = sgd.class_acc(mlp_c.forward(train_x), train_y)[0]
acc_test = sgd.class_acc(mlp_c.forward(test_x), test_y)[0]
print "Amazon Sentiment Accuracy train: %f test: %f\n" % (acc_train, acc_test)
