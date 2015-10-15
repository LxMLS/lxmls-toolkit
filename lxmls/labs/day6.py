'''
Draft from the second deep learning day
'''

import sys
sys.path.append('.')
import time

# FOR DEBUGGING
from ipdb import set_trace

print "\n######################",
print "\n   Exercise 6.1"
print "######################"

# Convince yourself a RNN is just an MLP with inputs and outputs at various
# layers. 

# TODO: Implement a Numpy RNN

print "\n######################",
print "\n   Exercise 6.2"
print "######################"

# Scan is your friend, maybe. Simple examples of scan, see some, IMPLEMENT some
# other. 

# TODO: Implement some examples covering typical caveats.

print "\n######################",
print "\n   Exercise 6.3"
print "######################"

# IMPLEMENT the numpy RNN in 6.1 with scan 

#
# Load POS and compacify the indices
#

import lxmls.readers.pos_corpus as pcc
corpus    = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",
                                            max_sent_len=15, max_nr_sent=1000)
test_seq  = corpus.read_sequence_list_conll("data/test-23.conll",
                                            max_sent_len=15, max_nr_sent=1000)
dev_seq   = corpus.read_sequence_list_conll("data/dev-22.conll",
                                            max_sent_len=15, max_nr_sent=1000)
# Redo indices so that they are consecutive
train_seq, test_seq, dev_seq = pcc.compacify(train_seq, test_seq, dev_seq)

#
# CREATE RNN TO PREDICT POS TAGS 
#

import numpy as np
import theano
import theano.tensor as T
import lxmls.deep_learning.rnn as rnn

# FORWARD
# CONFIG 
n_words = len(train_seq.x_dict.keys())           # Number of words
n_hidd  = 200                                    # Size of the recurrent layer
n_tags  = len(train_seq.y_dict.keys())           # Number of POS tags
# SYMBOLIC VARIABLES
_x      = T.ivector('x')    # Input words indices
_h0     = T.matrix('h0')    # Initial recurrent layer values
# Define the RNN
rnn     = rnn.RNN(n_words, n_hidd, n_tags)
# Symbolic forward
_p_y    = rnn._forward(_x, _h0)

# CLASSIFICATION COST
_y      = T.ivector('y')    # True output tags indices
_F      = -T.mean(T.log(_p_y)[T.arange(_y.shape[0]), _y])

# SGD UPDATE RULE
lrate   = 0.01
updates = [(par, par - lrate*T.grad(_F, par)) for par in rnn.param] 

# COMPILE FORWARD
fwd   = theano.function([_x, _h0], _p_y)
x     = np.array(train_seq[0].x).astype('int32')
y     = np.array(train_seq[0].y).astype('int32')
h0    = np.zeros((1, n_hidd)).astype(theano.config.floatX)

# COMPILE BATCH UPDATE
batch_update = theano.function([_x, _h0, _y], _F, updates=updates)

# DEVEL
err = 0
N   = 0
for n, seq in enumerate(dev_seq):
    x      = np.array(train_seq[0].x).astype('int32')
    y      = np.array(train_seq[0].y).astype('int32')
    h0     = np.zeros((1, n_hidd)).astype(theano.config.floatX)
    hat_y  = np.argmax(fwd(x, h0), 1)
    err    += sum(hat_y != y)
    N      += y.shape[0]
print "Acc %2.2f %%" % (100*(1 - err*1./N))

for i in range(10):

    # TRAIN EPOCH
    cost =0
    for n, seq in enumerate(train_seq):
        x      = np.array(train_seq[0].x).astype('int32')
        y      = np.array(train_seq[0].y).astype('int32')
        h0     = np.zeros((1, n_hidd)).astype(theano.config.floatX)
        cost += batch_update(x, h0, y)
        print "\r%d/%d cost %2.2f" % (n+1, len(train_seq), cost),
    
    # DEVEL
    err = 0
    N   = 0
    for n, seq in enumerate(dev_seq):
        x      = np.array(train_seq[0].x).astype('int32')
        y      = np.array(train_seq[0].y).astype('int32')
        h0     = np.zeros((1, n_hidd)).astype(theano.config.floatX)
        hat_y  = np.argmax(fwd(x, h0), 1)
        err    += sum(hat_y != y)
        N      += y.shape[0]
    print "Acc %2.2f %%" % (100*(1 - err*1./N))

set_trace()

print "\n######################",
print "\n   Exercise 6.4"
print "######################"

# Convince yourself that LSTMs and GRUs are just slightly more complex RNNs
# Use here those nice pics from the blog-post
