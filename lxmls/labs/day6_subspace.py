#!/usr/bin/python
import lxmls.deep_learning.SemEvalTwitter as ST
import numpy as np
import theano
from lxmls.deep_learning.embSubspace import embSubMLP

# DEBUGGING
from ipdb import set_trace


####################################
#           CONFIG 
####################################

DO_TRAIN = 1
# TRAIN CONFIGURATION
n_iter  = 6
lrate   = np.array(0.01).astype(theano.config.floatX)

# READ SEMEVAL DATA
sr = ST.SemEvalReader(one_hot=False)
dev_x, dev_y     = sr.dev
train_x, train_y = sr.train
t2015_x, t2015_y = sr.tweets2015
t2014_x, t2014_y = sr.tweets2014
t2013_x, t2013_y = sr.tweets2013

#reformat the labels
dev_y   = [np.array(dy).astype('int32')[None] for dy in dev_y]
train_y = [np.array(ty).astype('int32')[None] for ty in train_y]
t2015_y = [np.array(ty).astype('int32')[None] for ty in t2015_y]
t2014_y = [np.array(ty).astype('int32')[None] for ty in t2014_y]
t2013_y = [np.array(ty).astype('int32')[None] for ty in t2013_y]

# SUBSPACE EMBEDDING
subsize    = 10
model_path = 'data/twitter/features/semeval.pkl'

# Create or load model
if DO_TRAIN:
    #get the pretrained embeddings
    E = sr.get_embedding()
    E = E.astype(theano.config.floatX)
    nn  = embSubMLP(E, n_h1=subsize)
    print "Training %s" % model_path    
    nn.train((train_x,train_y), (dev_x,dev_y), lrate, n_iter)
else:
    nn = embSubMLP(None, model_file=model_path)

####################################
#              TEST
####################################

print "---TESTING---"
print "\nTrain"
Fmes_train, acc_train = nn.evaluate(train_x, train_y)
print "\nTweets 2013"
Fmes_t2013, acc_t2013 = nn.evaluate(t2013_x, t2013_y)
print "\nTweets 2014"
Fmes_t2014, acc_t2014 = nn.evaluate(t2014_x, t2014_y)
print "\nTweets 2015"
Fmes_t2015, acc_t2015 = nn.evaluate(t2015_x, t2015_y)

print "\n\nAvg. F-measure (POS/NEG)"
print "train: %.3f\ntweets 2015: %.3f\ntweets 2014: %.3f\ntweets 2013: %.3f\n"%(Fmes_train, Fmes_t2015, Fmes_t2014, Fmes_t2013)

print "\nAccuracy"
print "train: %.3f\ntweets 2015: %.3f\ntweets 2014: %.3f\ntweets 2013: %.3f\n"%(acc_train, acc_t2015, acc_t2014, acc_t2013)


