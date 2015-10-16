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
# Redo indices so that they are consecutive. Also cast all data to numpy arrays
# of int32 for compatibility with GPUs and theano.
train_seq, test_seq, dev_seq = pcc.compacify(train_seq, test_seq, dev_seq,
                                             theano=True)

#
# CREATE RNN TO PREDICT POS TAGS 
#

import numpy as np
import theano
import theano.tensor as T
import lxmls.deep_learning.rnn as rnn

# FORWARD
# CONFIG 
n_words = len(train_seq.x_dict.keys())     # Number of words
n_hidd  = 10                               # Size of the recurrent layer
n_tags  = len(train_seq.y_dict.keys())     # Number of POS tags
# SYMBOLIC VARIABLES
_x      = T.ivector('x')                   # Input words indices
_y      = T.ivector('y')                   # True output tags indices
# Define the RNN
rnn     = rnn.RNN(n_words, n_hidd, n_tags)
# Forward
_p_y    = rnn._forward(_x)
# Train cost
_F      = -T.mean(T.log(_p_y)[T.arange(_y.shape[0]), _y]) 
# Total prediction error 
_err    = T.sum(T.neq(T.argmax(_p_y,1), _y))
err_sum = theano.function([_x, _y], _err)

# SGD UPDATE RULE
lrate   = 0.5
updates = [(_par, _par - lrate*T.grad(_F, _par)) for _par in rnn.param] 

# COMPILE FORWARD, BATCH UPDATE
fwd          = theano.function([_x], _p_y)
batch_update = theano.function([_x, _y], _F, updates=updates)

# DEVEL
err = 0
N   = 0
for n, seq in enumerate(dev_seq):
#    x      = np.array(train_seq[0].x).astype('int32')
#    y      = np.array(train_seq[0].y).astype('int32')
    err  += err_sum(seq.x, seq.y)
    N     += seq.y.shape[0]
print "Acc %2.2f %%" % (100*(1 - err*1./N))

for i in range(20):

    # TRAIN EPOCH
    cost = 0
    err  = 0
    N    = 0
    for n, seq in enumerate(train_seq):
#        x     = np.array(seq.x).astype('int32')
#        y     = np.array(seq.y).astype('int32')
        err  += err_sum(seq.x, seq.y)
        N    += seq.x.shape[0]
        cost += batch_update(seq.x, seq.y)
        perc  = (n+1)*100./len(train_seq) 
        sys.stdout.write("\r%2.2f %%" % perc)
        sys.stdout.flush()

    Acc = (1 - err*1./N)*100.
    sys.stdout.write("\rTrain %d/%d cost %2.2f Acc %2.2f %%"  
                     % (n+1, len(train_seq), cost, Acc))
    sys.stdout.flush()
    print ""
    
    # DEVEL
    err = 0
    N   = 0
    for n, seq in enumerate(dev_seq):
#        x    = np.array(seq.x).astype('int32')
#        y    = np.array(seq.y).astype('int32')
        err  += err_sum(seq.x, seq.y)
        N   += seq.y.shape[0]
    print "Devel Acc %2.2f %%" % (100*(1 - err*1./N))

# TEST 
err = 0
N   = 0
for n, seq in enumerate(test_seq):
#    x      = np.array(seq.x).astype('int32')
#    y      = np.array(seq.y).astype('int32')
    err   += err_sum(seq.x, seq.y)
    N     += seq.y.shape[0]
print "Test Acc %2.2f %%" % (100*(1 - err*1./N))

set_trace()

print "\n######################",
print "\n   Exercise 6.4"
print "######################"

# Convince yourself that LSTMs and GRUs are just slightly more complex RNNs
# Use here those nice pics from the blog-post
