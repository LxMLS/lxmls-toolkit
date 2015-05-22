#!/usr/bin/python
'''
Deep learning day exercises corresponding to Chapter 7 of the LxMLS guide.
They must be run from ./lxmls-toolkit/ folder as

./lxmls/labs/day7.py

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
'''

import sys
sys.path.append('.')
import time

print "\n######################",
print "\n   Exercise 7.1"
print "######################"
import numpy as np
import lxmls.readers.sentiment_reader as srs  
scr       = srs.SentimentCorpus("books")
train_set = (scr.train_X.T, scr.train_y[:,0].astype(np.int32))
test_set  = (scr.test_X.T, scr.test_y[:,0].astype(np.int32))
###
import lxmls.deep_learning.mlp as dl
import lxmls.deep_learning.sgd as sgd
I = train_set[0].shape[0] 
mlp = dl.MLP(geometry=(I, 2), actvfunc=['softmax'])
#
sgd.SGD_train(mlp, train_set=train_set, batch_size=5, n_iter=5)
acc_train = sgd.class_acc(mlp.forward(train_set[0]), train_set[1])[0]
acc_test  = sgd.class_acc(mlp.forward(test_set[0]), test_set[1])[0]
print "Log-linear Model Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)

print "\n######################",
print "\n   Exercise 7.2"
print "######################"
geometry=(I, 20, 2)
mlp = dl.MLP(geometry=geometry)
sgd.SGD_train(mlp, train_set=train_set, batch_size=5, n_iter=5)
acc_train = sgd.class_acc(mlp.forward(train_set[0]), train_set[1])[0]
acc_test  = sgd.class_acc(mlp.forward(test_set[0]), test_set[1])[0]
print "MLP %s Model Amazon Sentiment Accuracy train: %f test: %f"%(geometry, acc_train,acc_test)


print "\n######################",
print "\n   Exercise 7.3"
print "######################"
import numpy as np
x        = test_set[0]        # Test set 
W1, w1   = mlp.weights[0]     # Weigths and bias of fist layer 
z1       = np.dot(W1, x) + w1 # Linear transformation
tilde_z1 = 1/(1+np.exp(-z1))  # Non-linear transformation  
#
import theano
import theano.tensor as T
symb_x = T.matrix('x')
symb_W1 = theano.shared(value=W1, name='W1', borrow=True)
symb_w1 = theano.shared(value=w1, name='w1', borrow=True, broadcastable=(False, True)) 
#
symb_z1       = T.dot(symb_W1, symb_x) + symb_w1
symb_tilde_z1 = T.nnet.sigmoid(symb_z1)
#
layer1 = theano.function([symb_x], symb_tilde_z1)

print ""
print "These two should be the same"
print ""
print tilde_z1
print ""
print layer1(x)


print "\n######################",
print "\n   Exercise 7.4"
print "######################"
mlp_a = dl.MLP(geometry=geometry)
mlp_b = dl.TheanoMLP(geometry=geometry)
#
print ""
print "These two should be the same"
print ""
print mlp_a.forward(test_set[0])[:,:10] 
print ""
print mlp_b.forward(test_set[0])[:,:10]


print "\n######################",
print "\n   Exercise 7.5"
print "######################"
W2, w2 = mlp.weights[1] # Weigths and bias of second (and last!) layer
# Second layer symbolic variables
symb_W2 = theano.shared(value=W2, name='W2', borrow=True)
symb_w2 = theano.shared(value=w2, name='w2', borrow=True, broadcastable=(False, True)) # Second layer symbolic expressions
symb_z2       = T.dot(symb_W2, symb_tilde_z1) + symb_w2
symb_tilde_z2 = T.nnet.softmax(symb_z2.T)
#
symb_y = T.ivector('y')
#
symb_F = -T.sum(T.log(symb_tilde_z2[T.arange(symb_y.shape[0]), symb_y]))
#
symb_nabla_F = T.grad(symb_F, symb_W1)
nabla_F      = theano.function([symb_x, symb_y], symb_nabla_F)

print "\n######################",
print "\n   Exercise 7.6"
print "######################"
geometry=(I, 20, 2)
mlp1 = dl.MLP(geometry=geometry)
mlp2 = dl.TheanoMLP(geometry=geometry)
mlp3 = dl.TheanoMLP(geometry=geometry)
#
mlp3.compile_train(train_set=train_set, batch_size=5)
#
init_t = time.clock()
sgd.SGD_train(mlp1, train_set=train_set, batch_size=5, n_iter=5)
print "\nNumpy version took %2.2f\n" % (time.clock() - init_t)
init_t = time.clock()
sgd.SGD_train(mlp2, train_set=train_set, batch_size=5, n_iter=5)
print "\nCompiled gradient version took %2.2f\n" % (time.clock() - init_t)
init_t = time.clock()
sgd.SGD_train(mlp3, n_iter=5)
print "\nTheano compiled batch update version took %2.2f\n" % (time.clock() - init_t)
init_t = time.clock()
