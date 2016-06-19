'''
Draft from the second deep learning day
'''

import sys
sys.path.append('.')
import os
import time

# FOR DEBUGGING
from ipdb import set_trace

print "\n######################",
print "\n   Exercise 6.1"
print "######################"

# Convince yourself a RNN is just an MLP with inputs and outputs at various
# layers. 

# LOAD DATA    
import lxmls.readers.pos_corpus as pcc
corpus    = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",
                                            max_sent_len=15, max_nr_sent=1000)
test_seq  = corpus.read_sequence_list_conll("data/test-23.conll",
                                            max_sent_len=15, max_nr_sent=1000)
dev_seq   = corpus.read_sequence_list_conll("data/dev-22.conll",
                                            max_sent_len=15, max_nr_sent=1000)
import lxmls.deep_learning.embeddings as emb 
if not os.path.isfile('data/senna_50'):
    emb.download_embeddings('senna_50', 'data/senna_50')
E = emb.extract_embeddings('data/senna_50', train_seq.x_dict)

# CONFIG 
n_words = E.shape[0]                        # Number of words
n_emb   = E.shape[1]                        # Size of word embeddings
n_hidd  = 20                                # Size of the recurrent layer
n_tags  = len(train_seq.y_dict.keys())      # Number of POS tags
seed = 0                                    # seed to initalize rnn parameters

# Test NumpyRNN() with the sample=0
sample = 0                  # sample to be tested
x0 = train_seq[sample].x    # first sample input (vector of integers)
y0 = train_seq[sample].y    # first sample output (vector of integers)

import lxmls.deep_learning.rnn as rnns
np_rnn = rnns.NumpyRNN(E, n_hidd, n_tags)
loos, p_y, p, y_rnn, h, z1, x = np_rnn.forward(x0, allOuts=True, outputs=y0)
nabla_params = np_rnn.grads(x0, y0)

set_trace()

# Save loss and gradient to compare with theano output 
numpy_loos = loos
numpy_grads = nabla_params

# CONFIG
lrate   = 0.5  # Learning rate          
n_iter  = 20   # Number of iterations

#
# TRAIN MODEL WITH SGD
#


print "\n######################",
print "\n   Exercise 6.2"
print "######################"

# Scan is your friend, maybe. Simple examples of scan, see some, IMPLEMENT some
# other. 

# TODO: Implement some examples covering typical caveats.

print "\n######################",
print "\n   Exercise 6.3"
print "######################"

import theano
import theano.tensor as T

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
import lxmls.deep_learning.rnn as rnns

#
# DEFINE MODEL
#

# Extract word embeddings for the vocabulary used. Download embeddings if
# not available.
import os
if not os.path.isfile('data/senna_50'):
    rnns.download_embeddings('senna_50','data/senna_50')
E = rnns.extract_embeddings('data/senna_50', train_seq.x_dict)

# CONFIG 
n_words = E.shape[0]                        # Number of words
n_emb   = E.shape[1]                        # Size of word embeddings
n_hidd  = 20                                # Size of the recurrent layer
n_tags  = len(train_seq.y_dict.keys())      # Number of POS tags
# SYMBOLIC VARIABLES
_x      = T.ivector('x')                    # Input words indices
# Define the RNN
rnn     = rnns.RNN(E, n_hidd, n_tags, seed=seed)
# Forward
_p_y    = rnn._forward(_x)

#
# DEFINE TRAINING 
#

# CONFIG
lrate   = 0.5  # Learning rate          
n_iter  = 20   # Number of iterations
# SYMBOLIC VARIABLES
_y      = T.ivector('y')                   # True output tags indices
# Train cost
_F      = -T.mean(T.log(_p_y)[T.arange(_y.shape[0]), _y]) 
# Total prediction error 
_err    = T.sum(T.neq(T.argmax(_p_y,1), _y))

# SGD UPDATE RULE
updates = [(_par, _par - lrate*T.grad(_F, _par)) for _par in rnn.param] 

# COMPILE ERROR FUNCTION, BATCH UPDATE
err_sum      = theano.function([_x, _y], _err)
batch_update = theano.function([_x, _y], _F, updates=updates)

## ADDED by MLA: Comparison with NumpyRNN
import numpy as np
grads_func  = theano.function([_x, _y], [T.grad(_F, _par) for _par in rnn.param])
F_func      = theano.function([_x, _y], _F)
F           = F_func(x0, y0)
grads       = grads_func(x0, y0)

grads2 = np_rnn.grads(x0, y0)

set_trace()

print 'Difference in Loss:', numpy_loos - F
print 'Difference in gradients:'
for ii, grad in enumerate(grads):

    try:
        assert np.allclose(numpy_grads[ii], grads[ii]), \
            "Numpy/Theano grads do not match"
    except:    
        set_trace()
        print ""

## Done comparison

#
# TRAIN MODEL WITH SGD
#

# Function computing accuracy for a sequence of sentences
def accuracy(seq):
    err = 0
    N   = 0
    for n, seq in enumerate(seq):
        err += err_sum(seq.x, seq.y)
        N   += seq.y.shape[0]
    return 100*(1 - err*1./N) 

print "\nTraining RNN for POS"
# EPOCH LOOP
for i in range(n_iter):

    # SENTENCE LOOP
    cost = 0
    for n, seq in enumerate(train_seq):
        cost += batch_update(seq.x, seq.y)
        # INFO
        perc  = (n+1)*100./len(train_seq) 
        sys.stdout.write("\r%2.2f %%" % perc)
        sys.stdout.flush()

    # Accuracy on train and dev set
    Acc = accuracy(train_seq)
    print "\rEpoch %d: Train cost %2.2f Acc %2.2f %%" % (i+1, cost, Acc),
    print " Devel Acc %2.2f %%" % accuracy(dev_seq)

# Final accuracy on the dev set
print "Test Acc %2.2f %%" % accuracy(test_seq)

print "\n######################",
print "\n   Exercise 6.4"
print "######################"

# Convince yourself that LSTMs and GRUs are just slightly more complex RNNs
# TODO: Use here those nice pics from the blog-post

# Define the LSTM
lstm = rnns.LSTM(E, n_hidd, n_tags)
# Forward
_p_y = lstm._forward(_x)
# Train cost
_F   = -T.mean(T.log(_p_y)[T.arange(_y.shape[0]), _y]) 
# Total prediction error 
_err = T.sum(T.neq(T.argmax(_p_y,1), _y))

# SGD UPDATE RULE
updates = [(_par, _par - lrate*T.grad(_F, _par)) for _par in lstm.param] 

# COMPILE ERROR FUNCTION, BATCH UPDATE
err_sum      = theano.function([_x, _y], _err)
batch_update = theano.function([_x, _y], _F, updates=updates)

print "\nTraining LSTM for POS"
# EPOCH LOOP
for i in range(n_iter):
    cost = 0
    for n, seq in enumerate(train_seq):
        cost += batch_update(seq.x, seq.y)
        # INFO
        perc  = (n+1)*100./len(train_seq) 
        sys.stdout.write("\r%2.2f %%" % perc)
        sys.stdout.flush()

    # Accuracy on train and dev set
    Acc = accuracy(train_seq)
    print "\rEpoch %d: Train cost %2.2f Acc %2.2f %%" % (i+1, cost, Acc),
    print " Devel Acc %2.2f %%" % accuracy(dev_seq)

# Final accuracy on the dev set
print "Test Acc %2.2f %%" % accuracy(test_seq)
