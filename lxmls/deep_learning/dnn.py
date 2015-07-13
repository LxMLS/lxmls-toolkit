'''
SemEval models
'''

import sys
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import cPickle

def init_W(size, rng):
    '''
    Random initialization
    '''
    if len(size) == 2:
        n_out, n_in = size
    else:
        n_out, n_in = size[0], size[3]
    w0 = np.sqrt(6./(n_in + n_out))   
    W = np.asarray(rng.uniform(low=-w0, high=w0, size=size))
    return theano.shared(W.astype(theano.config.floatX), borrow=True)

class embLogLinear2():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, n_h1=20, model_file=None, hidden=False):

        self.hidden = hidden 

        # Random Seed
        rng = np.random.RandomState(1234)
        lex_feat_size = 80
        if model_file:

            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [W1, W2, W3] = cPickle.load(fid)
            W1 = theano.shared(W1, borrow=True)
            W2 = theano.shared(W2, borrow=True)
            W3 = theano.shared(W3, borrow=True)

        else:

            # Embeddings e.g. Wang's, word2vec.   
            with open(emb_path, 'rb') as fid:
                W1 = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = W1.shape
            # This is fixed!
            W1 = theano.shared(W1, borrow=True)
            # Embedding subspace projection
            W2 = init_W((n_h1, emb_size), rng) 
            # Hidden layer
            # n_h2 = n_h1+lex_feat_size
            # W_emb_lex = init_W((n_h2, n_h1), rng)
            # Hidden layer
            W3 = init_W((3, n_h1), rng) 

        # Fixed parameters
        self.W1     = W1
        # Parameters to be updated 
        self.params = [W2, W3]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def gradients(self, x, y):
        return [gr(x.astype('int32'), y.astype('int32')) for gr in self.grads]

    def compile(self):
        '''
        Forward pass and Gradients
        '''
       
        # Get nicer names for parameters
        W1, W2, W3 = [self.W1] + self.params

        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()       # tweet in one hot

        # Use an intermediate sigmoid
	if self.hidden == True:
            z1b = W1[:, self.z0]       # embedding
            z1  = T.nnet.sigmoid(z1b) 

        else:
            z1  = W1[:, self.z0]       # embedding
        z2         = T.dot(W2, z1)  # subspace
        # Hidden layer
        z3         = T.dot(W3, z2)
        z4         = T.sum(z3, 1)                   # Bag of words
        self.hat_y = T.nnet.softmax(z4.T).T
        self.fwd   = theano.function([self.z0], self.hat_y)
        
        # TRAINING COST AND GRADIENTS
        # Train cost minus log probability
        self.y = T.ivector()                     # reference out
        self.F = -T.mean(T.log(self.hat_y)[self.y])   # For softmax out 
        # Update only last three parameters
        self.nablas = [] # Symbolic gradients
        self.grads  = [] # gradients
        for W in self.params:
            self.nablas.append(T.grad(self.F, W))
            self.grads.append(theano.function([self.z0, self.y], T.grad(self.F, W)))
        self.cost = theano.function([self.z0, self.y], self.F)

    def save(self, model_file):
        with open(model_file, 'wb') as fid: 
            param_list = [self.W1.get_value()] + [W.get_value() 
                          for W in self.params]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)


class embLogLinear():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, model_file=None):

        # Random Seed
        rng = np.random.RandomState(1234)
        if model_file:

            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [W1, W2] = cPickle.load(fid)
            W1 = theano.shared(W1, borrow=True)
            W2 = theano.shared(W2, borrow=True)

        else:

            # Embeddings e.g. Wang's, word2vec.   
            with open(emb_path, 'rb') as fid:
                W1 = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = W1.shape
            # This is fixed!
            W1 = theano.shared(W1, borrow=True)
            # Embedding subspace projection
            W2 = init_W((3, emb_size), rng) 

        # Fixed parameters
        self.W1     = W1
        # Parameters to be updated 
        self.params = [W2]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def gradients(self, x, y):
        return [gr(x.astype('int32'), y.astype('int32')) for gr in self.grads]

    def compile(self):
        '''
        Forward pass and Gradients
        '''
       
        # Get nicer names for parameters
        W1, W2 = [self.W1] + self.params

        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()              # tweet in one hot
        z1         = W1[:, self.z0]           # embedding
        z2         = T.sum(T.dot(W2, z1), 1)  # subspace
        self.hat_y = T.nnet.softmax(z2.T).T
        self.fwd   = theano.function([self.z0], self.hat_y)
        
#        train_data = 'DATA/features/wang/semeval-pretokenized.pkl'
#        # SEMEVAL TRAIN TWEETS
#        print "Loading %s" % train_data 
#        with open(train_data, 'rb') as fid:
#            [word2idx, idx2wrd, train_x, train_y, dev_x, dev_y] = cPickle.load(fid) 
#        f2 = theano.function([self.z0], z2)
#        import ipdb;ipdb.set_trace()

        # TRAINING COST AND GRADIENTS
        # Train cost minus log probability
        self.y = T.ivector()                          # reference out
        self.F = -T.mean(T.log(self.hat_y)[self.y])   # For softmax out 
        # Update only last three parameters
        self.nablas = [] # Symbolic gradients
        self.grads  = [] # gradients
        for W in self.params:
            self.nablas.append(T.grad(self.F, W))
            self.grads.append(theano.function([self.z0, self.y], T.grad(self.F, W)))
        self.cost = theano.function([self.z0, self.y], self.F)

    def save(self, model_file):
        with open(model_file, 'wb') as fid: 
            param_list = [self.W1.get_value()] + [W.get_value() 
                          for W in self.params]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)



class embSubMLP():
    '''
    Embedding subspace
    '''
    def __init__(self, emb_path, n_h1=20, model_file=None, hidden=False):

        self.hidden = hidden 

        # Random Seed
        rng = np.random.RandomState(1234)
        lex_feat_size = 80
        if model_file:

            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [W1, W2, W3] = cPickle.load(fid)
            W1 = theano.shared(W1, borrow=True)
            W2 = theano.shared(W2, borrow=True)
            W3 = theano.shared(W3, borrow=True)

        else:

            # Embeddings e.g. Wang's, word2vec.   
            with open(emb_path, 'rb') as fid:
                W1 = cPickle.load(fid).astype(theano.config.floatX)
            emb_size, voc_size = W1.shape
            # This is fixed!
            W1 = theano.shared(W1, borrow=True)
            # Embedding subspace projection
            W2 = init_W((n_h1, emb_size), rng) 
            # Hidden layer
            # n_h2 = n_h1+lex_feat_size
            # W_emb_lex = init_W((n_h2, n_h1), rng)
            # Hidden layer
            W3 = init_W((3, n_h1), rng) 

        # Fixed parameters
        self.W1     = W1
        # Parameters to be updated 
        self.params = [W2, W3]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def gradients(self, x, y):
        return [gr(x.astype('int32'), y.astype('int32')) for gr in self.grads]

    def compile(self):
        '''
        Forward pass and Gradients
        '''
       
        # Get nicer names for parameters
        W1, W2, W3 = [self.W1] + self.params

        # FORWARD PASS
        # Embedding layer subspace
        self.z0    = T.ivector()                    # tweet in one hot

        # Use an intermediate sigmoid
	if self.hidden == True:

            z1b = W1[:, self.z0]                 # embedding
            z1  = T.nnet.sigmoid(z1b) 

        else:
            z1  = W1[:, self.z0]                 # embedding


        z2         = T.nnet.sigmoid(T.dot(W2, z1))  # subspace
        # Hidden layer
        z3         = T.dot(W3, z2)
        z4         = T.sum(z3, 1)                   # Bag of words
        self.hat_y = T.nnet.softmax(z4.T).T
        self.fwd   = theano.function([self.z0], self.hat_y)
        
        # TRAINING COST AND GRADIENTS
        # Train cost minus log probability
        self.y = T.ivector()                          # reference out
        self.F = -T.mean(T.log(self.hat_y)[self.y])   # For softmax out 
        # Update only last three parameters
        self.nablas = [] # Symbolic gradients
        self.grads  = [] # gradients
        for W in self.params:
            self.nablas.append(T.grad(self.F, W))
            self.grads.append(theano.function([self.z0, self.y], T.grad(self.F, W)))
        self.cost = theano.function([self.z0, self.y], self.F)

    def save(self, model_file):
        with open(model_file, 'wb') as fid: 
            param_list = [self.W1.get_value()] + [W.get_value() 
                          for W in self.params]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)

class logLinear():
    '''
    Log linear
    '''
    def __init__(self, voc_size, model_file=None, unk2null=False, pos_W=False):

        # Random Seed
        rng = np.random.RandomState(1234)

        self.unk2null = unk2null
        self.pos_W    = pos_W

        if model_file:

            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                W1 = cPickle.load(fid)
            W1 = theano.shared(W1, borrow=True)

        else:

            # Embedding subspace projection
            W1 = init_W(voc_size, 3, rng) 

        # Parameters to be updated 
        self.params = [W1]
        # Compile
        self.compile()

    def forward(self, x):
        return self.fwd(x.astype('int32'))

    def gradients(self, x, y):
        return [gr(x.astype('int32'), y.astype('int32')) for gr in self.grads]

    def compile(self):
        '''
        Forward pass and Gradients
        '''
        # Get nicer names for parameters
        W1 = self.params[0]
        # FORWARD PASS
        # Embedding layer subspace 
        self.z0    = T.ivector()                    # tweet in one hot               
        if self.pos_W:
            # Force embedding to be a positive number
            W1b = T.exp(W1)
            z1  = W1b[:, self.z0]                 # embedding
        else:
            z1  = W1[:, self.z0]                 # embedding
        z2         = T.sum(z1, 1)                   # Bag of words
        self.hat_y = T.nnet.softmax(z2.T).T
        self.fwd   = theano.function([self.z0], self.hat_y)
        
        # TRAINING COST AND GRADIENTS
        # Train cost minus log probability
        self.y = T.ivector()                     # reference out
        self.F = -T.mean(T.log(self.hat_y)[self.y])   # For softmax out 
        # Update only last three parameters
        self.nablas = [] # Symbolic gradients
        self.grads  = [] # gradients

        if self.unk2null:
            #import ipdb;ipdb.set_trace()
            # Mask gradient for the unknown word so that it is not updated
            M        = np.ones(W1.get_value().shape)
            M[:, :2] = 0
            # Set indices 0 and 1 to zero as they are reserved words for 
            # unknowns
            W1.set_value(W1.get_value()*M)
            #
            self.M    = theano.shared(M.astype(theano.config.floatX), borrow=True)
            self.nablas.append(T.grad(self.F, W1) * M)

        self.nablas.append(T.grad(self.F, W1))
        self.grads.append(theano.function([self.z0, self.y], T.grad(self.F, W1)))
        self.cost = theano.function([self.z0, self.y], self.F)

    def save(self, model_file):
        W1 = self.params[0]
        with open(model_file, 'wb') as fid: 
            cPickle.dump(W1.get_value(), fid, cPickle.HIGHEST_PROTOCOL)
