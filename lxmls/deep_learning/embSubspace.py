'''
SemEval models
'''

import cPickle
from ipdb import set_trace
import numpy as np
import theano
import sys
import theano.tensor as T
import SemEvalTwitter as ST

model_path = 'data/twitter/features/semeval.pkl'

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

class embSubMLP():
    '''
    Embedding subspace
    '''
    def __init__(self, E, n_h1=20, model_file=None, hidden=False):

        self.hidden = hidden 

        # Random Seed
        rng = np.random.RandomState(1234)        
        if model_file:
            # Load pre existing model  
            with open(model_file, 'rb') as fid: 
                [E, S, Y] = cPickle.load(fid)
            E = theano.shared(E, borrow=True)
            S = theano.shared(S, borrow=True)
            Y = theano.shared(Y, borrow=True)
        else:                        
            emb_size, voc_size = E.shape
            # This is fixed!
            E = theano.shared(E, borrow=True)
            # Embedding subspace projection
            S = init_W((n_h1, emb_size), rng)             
            # Hidden layer
            Y = init_W((3, n_h1), rng) 
        # Fixed parameters
        self.E     = E
        # Parameters to be updated 
        self.params = [S, Y]
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
        
        E, S, Y = [self.E] + self.params
        # FORWARD PASS
        # Embedding layer subspace
        self.z0 = T.ivector()                   # tweet in one hot
        # Use an intermediate sigmoid
        if self.hidden == True:
            z1b = E[:, self.z0]                   # embedding
            z1  = T.nnet.sigmoid(z1b) 
        else:
            z1  = E[:, self.z0]                   # embedding
        z2         = T.nnet.sigmoid(T.dot(S, z1)) # subspace
        # Hidden layer
        z3         = T.dot(Y, z2)
        z4         = T.sum(z3, 1)                  # Bag of words
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
            param_list = [self.E.get_value()] + [W.get_value() 
                          for W in self.params]
            cPickle.dump(param_list, fid, cPickle.HIGHEST_PROTOCOL)

    def evaluate(self, eval_x, eval_y):

        acc     = 0.
        mapp    = np.array([ 1, 2, 0])
        conf_mat = np.zeros((3, 3))
        for j, x, y in zip(np.arange(len(eval_x)), eval_x, eval_y):
            # Prediction
            p_y   = self.forward(x)
            hat_y = np.argmax(p_y)
            # Confusion matrix
            conf_mat[mapp[y[0]], mapp[hat_y]] += 1
            # Accuracy
            acc    = (acc*j + (hat_y == y[0]).astype(float))/(j+1)

            # INFO
            sys.stdout.write("\rTest %d/%d " % (j+1, len(eval_x)))
            sys.stdout.flush()   
        # set_trace()
        return ST.FmesSemEval(confusionMatrix=conf_mat), acc

    def train(self, train, dev, lrate, n_iter):
        
        train_x = train[0]
        train_y = train[1]
        dev_x   = dev[0]
        dev_y   = dev[1]
        
        # RESHAPE TRAIN DATA AS A SINGLE NUMPY ARRAY
        # Start and end indices
        lens = np.array([len(tr) for tr in train_x]).astype('int32')
        st   = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype('int32')
        ed   = (st + lens).astype('int32')
        x    = np.zeros((ed[-1], 1))
        for i, ins_x in enumerate(train_x):        
            x[st[i]:ed[i]] = ins_x[:, None].astype('int32')         
        
        # Train data and instance start and ends
        x  = theano.shared(x.astype('int32'), borrow=True) 
        y  = theano.shared(np.array(train_y).astype('int32'), borrow=True)
        st = theano.shared(st, borrow=True)
        ed = theano.shared(ed, borrow=True)

        # Update rule
        updates = [(pr, pr-lrate*gr) for pr, gr in zip(self.params, self.nablas)]
        # Mini-batch
        i  = T.lscalar()
        givens={ self.z0 : x[st[i]:ed[i], 0],
                 self.y  : y[i] }
        train_batch = theano.function(inputs=[i], outputs=self.F, updates=updates, givens=givens)

        # Epoch loop
        last_cr  = None
        best_cr  = [0, 0]
        for i in np.arange(n_iter):
            # Training Epoch                         
            p_train = 0 
            for j in np.arange(len(train_x)).astype('int32'): 
                p_train += train_batch(j) 
                # DEBUG
                if np.any(np.isnan(self.params[0].get_value())):
                    set_trace()
                    print ""
                # INFO
                if not (j % 100):
                    sys.stdout.write("\rTraining %d/%d" % (j+1, len(train_x)))
                    sys.stdout.flush()   

            Fm, cr = self.evaluate(dev_x, dev_y)            
            # INFO
            if last_cr:
                # Keep bet model
                if best_cr[0] < cr:
                    best_cr = [cr, i+1]
                delta_cr = cr - last_cr
                if delta_cr >= 0:
                    print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[32m+%2.5f\033[0m (Fm %2.5f%%)" % 
                           (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
                else: 
                    print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[31m%2.5f\033[0m (Fm %2.5f%%)" % 
                           (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
            else:
                print "\rEpoch %2d/%2d: %2.5f (Fm %2.5f%%)" % (i+1, n_iter, cr*100,
                                                               Fm*100)
                best_cr = [cr, i+1]
            last_cr = cr

        # SAVE MODEL
        self.save(model_path)
