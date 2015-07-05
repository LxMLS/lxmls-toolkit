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
t2015_y = sr.tweets2015[1]
t2014_y = sr.tweets2014[1]
t2013_y = sr.tweets2013[1]

train_x = sr.get_bow("train")
t2015_x  = sr.get_bow("tweets2015")
t2014_x  = sr.get_bow("tweets2014")
t2013_x  = sr.get_bow("tweets2013")
dev_x = sr.get_bow("dev") 
dev_y = sr.dev[1]

train_x = train_x.astype(theano.config.floatX)
train_y = train_y.astype('int32')
t2015_x = t2015_x.astype(theano.config.floatX)
t2015_y = t2015_y.astype('int32')
t2014_x = t2014_x.astype(theano.config.floatX)
t2014_y = t2014_y.astype('int32')
t2013_x = t2013_x.astype(theano.config.floatX)
t2013_y = t2013_y.astype('int32')

# Store data as shared variables
# NOTE: This will push the data into the GPU memory when used
_train_x = theano.shared(train_x, 'train_x', borrow=True)
_train_y = theano.shared(train_y, 'train_y', borrow=True)

# MODEL PARAMETERS
n_class = len(np.unique(train_y))
I = train_x.shape[0]
geometry = [I, 600, 10, n_class]
actvfunc = ['sigmoid', 'sigmoid', 'softmax'] 
n_iter = 6
bsize  = 10
lrate  = 0.01

print "\n######################",
print "\n        RANDOM WEIGHTS"
print "######################\n"

# Model
mlp_c     = dl.TheanoMLP(geometry, actvfunc)
_x      = T.matrix('x')
_y      = T.ivector('y')
_F      = mlp_c._cost(_x, _y)
_j      = T.lscalar()
updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_c.params]
givens  = { _x : _train_x[:, _j*bsize:(_j+1)*bsize], 
            _y : _train_y[_j*bsize:(_j+1)*bsize] }
batch_up = theano.function([_j], _F, updates=updates, givens=givens)
n_batch  = train_x.shape[1]/bsize  + 1
init_t = time.clock()
sgd.SGD_train(mlp_c, n_iter, batch_up=batch_up, n_batch=n_batch, devel_set=(dev_x,dev_y))
print "\nTheano compiled batch update version took %2.2f sec" % (time.clock() - init_t)
init_t = time.clock()

#Compute predictions
train_pred = np.argmax(mlp_c.forward(train_x),0)
t2015_pred  = np.argmax(mlp_c.forward(t2015_x),0)
t2014_pred  = np.argmax(mlp_c.forward(t2014_x),0)
t2013_pred  = np.argmax(mlp_c.forward(t2013_x),0)
Fmes_train = ST.FmesSemEval(train_y, train_pred )
Fmes_t2015  = ST.FmesSemEval(t2015_y, t2015_pred)
Fmes_t2014  = ST.FmesSemEval(t2014_y, t2014_pred)
Fmes_t2013  = ST.FmesSemEval(t2013_y, t2013_pred)

print "Twitter Sentiment Analysis POS/NEG F-measure"
print "train: %f\ntweets 2015: %f\ntweets 2014: %f\ntweets 2013: %f\n"%(Fmes_train,Fmes_t2014, Fmes_t2014, Fmes_t2013)

print "\n######################",
print "\n PRETRAINED EMBEDDINGS"
print "######################\n"

#Load pretrained embeddings
E = sr.get_embedding()
mlp_d     = dl.TheanoMLP(geometry, actvfunc)
mlp_d.params[0].set_value(E)
_x      = T.matrix('x')
_y      = T.ivector('y')
_F      = mlp_d._cost(_x, _y)
_j      = T.lscalar()
updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_d.params]
givens  = { _x : _train_x[:, _j*bsize:(_j+1)*bsize], 
            _y : _train_y[_j*bsize:(_j+1)*bsize] }
batch_up = theano.function([_j], _F, updates=updates, givens=givens)
n_batch  = train_x.shape[1]/bsize  + 1
init_t = time.clock()
sgd.SGD_train(mlp_d, n_iter, batch_up=batch_up, n_batch=n_batch, devel_set=(dev_x,dev_y))
print "\nTheano compiled batch update version took %2.2f sec" % (time.clock() - init_t)
init_t = time.clock()

#Compute predictions
train_pred = np.argmax(mlp_d.forward(train_x),0)
t2015_pred  = np.argmax(mlp_d.forward(t2015_x),0)
t2014_pred  = np.argmax(mlp_d.forward(t2014_x),0)
t2013_pred  = np.argmax(mlp_d.forward(t2013_x),0)
Fmes_train = ST.FmesSemEval(train_y, train_pred )
Fmes_t2015  = ST.FmesSemEval(t2015_y, t2015_pred)
Fmes_t2014  = ST.FmesSemEval(t2014_y, t2014_pred)
Fmes_t2013  = ST.FmesSemEval(t2013_y, t2013_pred)

print "Twitter Sentiment Analysis POS/NEG F-measure"
print "train: %f\ntweets 2015: %f\ntweets 2014: %f\ntweets 2013: %f\n"%(Fmes_train,Fmes_t2014, Fmes_t2014, Fmes_t2013)

print "\n###########################",
print "\n PRETRAINED EMBEDDINGS (FIXED)"
print "######################\n"
mlp_e     = dl.TheanoMLP(geometry, actvfunc)
#set the weights in the first layer with the pre-trained embeddings
mlp_e.params[0].set_value(E)
_x      = T.matrix('x')
_y      = T.ivector('y')
_F      = mlp_e._cost(_x, _y)
_j      = T.lscalar()
#the weights in the first layer are not updated
updates = [(par, par - lrate*T.grad(_F, par)) for par in mlp_e.params[1:]]
givens  = { _x : _train_x[:, _j*bsize:(_j+1)*bsize], 
            _y : _train_y[_j*bsize:(_j+1)*bsize] }
batch_up = theano.function([_j], _F, updates=updates, givens=givens)
n_batch  = train_x.shape[1]/bsize  + 1
init_t = time.clock()
sgd.SGD_train(mlp_e, n_iter, batch_up=batch_up, n_batch=n_batch, devel_set=(dev_x,dev_y))
print "\nTheano compiled batch update version took %2.2f sec" % (time.clock() - init_t)
init_t = time.clock()

#Compute predictions
train_pred = np.argmax(mlp_e.forward(train_x),0)
t2015_pred  = np.argmax(mlp_e.forward(t2015_x),0)
t2014_pred  = np.argmax(mlp_e.forward(t2014_x),0)
t2013_pred  = np.argmax(mlp_e.forward(t2013_x),0)
Fmes_train = ST.FmesSemEval(train_y, train_pred )
Fmes_t2015  = ST.FmesSemEval(t2015_y, t2015_pred)
Fmes_t2014  = ST.FmesSemEval(t2014_y, t2014_pred)
Fmes_t2013  = ST.FmesSemEval(t2013_y, t2013_pred)

print "Twitter Sentiment Analysis POS/NEG F-measure"
print "train:%f\n tweets 2015: %f\ntweets 2014: %f\ntweets 2013: %f\n"%(Fmes_train,Fmes_t2014, Fmes_t2014, Fmes_t2013)

