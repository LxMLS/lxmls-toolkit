from ipdb import set_trace
import lxmls.deep_learning.mlp as dl 
import lxmls.deep_learning.sgd as sgd
import lxmls.deep_learning.PTB as PTB
import lxmls.deep_learning.SemEvalTwitter as ST
import numpy as np
import theano
from theano import tensor as T
import time
import cPickle


# READ SEMEVAL DATA
sr = ST.SemEvalReader()
train_y = sr.train[1]
test_y = sr.tweets2014[1]
train_x = sr.get_bow("train")
test_x  = sr.get_bow("tweets2014")

train_x = train_x.astype(theano.config.floatX)
train_y = train_y.astype('int32')
test_x = test_x.astype(theano.config.floatX)
test_y = test_y.astype('int32')

# Store data as shared variables
# NOTE: This will push the data into the GPU memory when used
_train_x = theano.shared(train_x, 'train_x', borrow=True)
_train_y = theano.shared(train_y, 'train_y', borrow=True)

#Load pretrained embeddings
# E = sr.get_embeddings()

# MODEL PARAMETERS
n_class = len(np.unique(train_y))
I = train_x.shape[0]
geometry = [I, 200, 10, n_class]
actvfunc = ['sigmoid', 'sigmoid', 'softmax'] 
n_iter = 10
bsize  = 5
lrate  = 0.01

# Model
mlp_c     = dl.TheanoMLP(geometry, actvfunc)


# mlp_c.params[0].set_value(E)

# Define givens variables to be used in the batch update
# Get symbolic variables returning a mini-batch of data 

# Define updates variable. This is a list of gradient descent updates 
# The output is a list following theano.function updates parameter. This
# consists on a list of tuples with each parameter and update rule
_x      = T.matrix('x')
_y      = T.ivector('y')
_F      = mlp_c._cost(_x, _y)
updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_c.params]

# Givens maps input and target to a mini-batch of inputs and targets 
_j      = T.lscalar()
givens  = { _x : _train_x[:, _j*bsize:(_j+1)*bsize], 
            _y : _train_y[_j*bsize:(_j+1)*bsize] }

# Define the batch update function. This will return the cost of each batch
# and update the MLP parameters at the same time using updates
batch_up = theano.function([_j], _F, updates=updates, givens=givens)
n_batch  = train_x.shape[1]/bsize  + 1

init_t = time.clock()
sgd.SGD_train(mlp_c, n_iter, batch_up=batch_up, n_batch=n_batch)
print "\nTheano compiled batch update version took %2.2f sec" % (time.clock() - init_t)
init_t = time.clock()

#Compute predictions
train_pred = np.argmax(mlp_c.forward(train_x),0)
test_pred  = np.argmax(mlp_c.forward(test_x),0)
Fmes_train = ST.FmesSemEval(train_y, train_pred )
Fmes_test  = ST.FmesSemEval(test_y, test_pred)

print "Twitter Sentiment Analysis SemEval metric train: %f test: %f\n"%(Fmes_train,Fmes_test)
