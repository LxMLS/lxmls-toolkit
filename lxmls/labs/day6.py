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

def extract_embeddings(embedding_path):
    with open(embedding_path) as fid:
        for i, line in enumerate(fid.readlines()):
            # Initialize
            if i == 0:
                 N    = len(line.split()[1:])     
                 E    = np.random.uniform(size=(N, len(train_seq.x_dict)))
                 n    = 0
            word = line.split()[0].lower() 
            if word[0].upper() + word[1:] in train_seq.x_dict:
                idx        = train_seq.x_dict[word[0].upper() + word[1:]]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            elif word in train_seq.x_dict:
                idx        = train_seq.x_dict[word]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            print "\r%d/%d" % (n, len(train_seq.x_dict)),    
    print "Embeddings have %2f%% OOV" % ((1-n*1./len(train_seq.x_dict))*100)
    return E

# Extract embeddings for the sentences
import numpy as np
embedding_path = 'senna_50'
#embedding_path = '/ffs/tmp/ramon/cajon/NLSE.DATA/txt/wiki.sskipngram.600'
E              = extract_embeddings(embedding_path)

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

# CONFIG 
#n_words = E.shape[0]                       # Number of words
#n_emb   = E.shape[1]                       # Number of words
n_words = len(train_seq.x_dict)             # Number of words in vocabulary
n_emb   = 50                                # Size of the embedding layer
n_hidd  = 10                                # Size of the recurrent layer
n_tags  = len(train_seq.y_dict.keys())      # Number of POS tags
# SYMBOLIC VARIABLES
_x      = T.ivector('x')                    # Input words indices
# Define the RNN
rnn     = rnns.RNN(n_words, n_emb, n_hidd, n_tags)
# Forward
_p_y    = rnn._forward(_x)

# Set embeddings
rnn.param[0].set_value(E)

#
# DEFINE TRAINING 
#

# CONFIG
lrate   = 0.5  # Learning rate          
n_iter  = 30   # Number of iterations
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

# Function computing accuracy for a sequence of sentences
def accuracy(seq):
    err = 0
    N   = 0
    for n, seq in enumerate(seq):
        err += err_sum(seq.x, seq.y)
        N   += seq.y.shape[0]
    return 100*(1 - err*1./N) 

#
# TRAIN MODEL WITH SGD
#

#TODO: Merge with the other SGD

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

exit()

print "\n######################",
print "\n   Exercise 6.4"
print "######################"

# Convince yourself that LSTMs and GRUs are just slightly more complex RNNs
# TODO: Use here those nice pics from the blog-post

# Define the LSTM
lstm = rnns.LSTM(n_words, n_hidd, n_tags)
# Forward
_p_y = lstm._forward(_x)
# Train cost
_F      = -T.mean(T.log(_p_y)[T.arange(_y.shape[0]), _y]) 
# Total prediction error 
_err    = T.sum(T.neq(T.argmax(_p_y,1), _y))

# SGD UPDATE RULE
updates = [(_par, _par - lrate*T.grad(_F, _par)) for _par in lstm.param] 

# COMPILE ERROR FUNCTION, BATCH UPDATE
err_sum      = theano.function([_x, _y], _err)
batch_update = theano.function([_x, _y], _F, updates=updates)

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
