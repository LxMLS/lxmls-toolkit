#!/usr/bin/python

import os
import urllib2
import numpy as np
import theano
import theano.tensor as T

from ipdb import set_trace

def download_embeddings(embbeding_name, target_file):
    '''
    Downloads file through http with progress report
    
    Obtained in stack overflow:
    http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http
    -using-python
    '''
    
    # Embedding download URLs
    if embbeding_name == 'senna_50':
        # senna_50 embeddings
        source_url = 'http://lxmls.it.pt/2015/wp-content/uploads/2015/senna_50'
    else:
        raise ValueError, ("I do not have embeddings %s for download" 
                           % embbeding_name)

    target_file_name = os.path.basename('data/senna_50')
    u = urllib2.urlopen(source_url)
    with open(target_file, 'wb') as f:
        meta         = u.info()
        file_size    = int(meta.getheaders("Content-Length")[0])
        file_size_dl = 0
        block_sz     = 8192
        print "Downloading: %s Bytes: %s" % (target_file_name, file_size)
        while True:
            text_buffer = u.read(block_sz)
            if not text_buffer:
                break
            file_size_dl += len(text_buffer)
            f.write(text_buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, 
                                           file_size_dl*100./file_size)
            status = status + chr(8)*(len(status)+1)
            print status,
    print ""            

def extract_embeddings(embedding_path, word_dict):
    '''
    Given embeddings in text form and a word dictionary construct embedding
    matrix. Words with no embedding get initialized to random.
    '''

    with open(embedding_path) as fid:
        for i, line in enumerate(fid.readlines()):
            # Initialize
            if i == 0:
                 N    = len(line.split()[1:])     
                 E    = np.random.uniform(size=(N, len(word_dict)))
                 n    = 0
            word = line.split()[0].lower() 
            if word[0].upper() + word[1:] in word_dict:
                idx        = word_dict[word[0].upper() + word[1:]]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            elif word in word_dict:
                idx        = word_dict[word]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            print "\rGetting embeddings for the vocabulary %d/%d" % (n, len(word_dict)),    
    OOV_perc =  (1-n*1./len(word_dict))*100        
    print "\n%2.1f%% OOV, missing embeddings set to random" % OOV_perc
    return E


class RNN():

    def __init__(self, W_e, n_hidd, n_tags):
        '''
        E       numpy.array Word embeddings of size (n_emb, n_words)
        n_hidd  int         Size of the recurrent layer 
        n_tags  int         Total number of tags
        '''

        # Dimension of the embeddings
        n_emb = W_e.shape[0]

        # MODEL PARAMETERS
        W_x = np.random.uniform(size=(n_hidd, n_emb))   # Input layer 
        W_h = np.random.uniform(size=(n_hidd, n_hidd))  # Recurrent layer
        W_y = np.random.uniform(size=(n_tags, n_hidd))  # Output layer
        # Cast to theano GPU-compatible type
        W_e = W_e.astype(theano.config.floatX)
        W_x = W_x.astype(theano.config.floatX)
        W_h = W_h.astype(theano.config.floatX)
        W_y = W_y.astype(theano.config.floatX)
        # Store as shared parameters
        _W_e = theano.shared(W_e, borrow=True)
        _W_x = theano.shared(W_x, borrow=True)
        _W_h = theano.shared(W_h, borrow=True)
        _W_y = theano.shared(W_y, borrow=True)

        # Class variables
        self.n_hidd = n_hidd
        self.param  = [_W_e, _W_x, _W_h, _W_y]

    def _forward(self, _x, _h0=None):

        # Default initial hidden is allways set to zero
        if _h0 is None:
            h0  = np.zeros((1, self.n_hidd)).astype(theano.config.floatX)
            _h0 = theano.shared(h0, borrow=True)

        # COMPUTATION GRAPH

        # Get parameters in nice form
        _W_e, _W_x, _W_h, _W_y = self.param

        # NOTE: Since _x contains the indices rather than full one-hot vectors,
        # use _W_e[:, _x].T instead of T.dot(_x, _W_e.T)

        # ----------
        # Solution to Exercise 6.3 

        # Embedding layer 
        _z1 = _W_e[:, _x].T
    
        # This defines what to do at each step
        def rnn_step(_x_tm1, _h_tm1, _W_x, W_h):
            return T.nnet.sigmoid(T.dot(_x_tm1, _W_x.T) + T.dot(_h_tm1, W_h.T))
    
        # This creates the variable length computation graph (unrols the rnn)
        _h, updates = theano.scan(fn=rnn_step, 
                                  sequences=_z1, 
                                  outputs_info=dict(initial=_h0),
                                  non_sequences=[_W_x ,_W_h])
    
        # Remove intermediate empty dimension
        _z2 = _h[:,0,:]
    
        # End of solution to Exercise 6.3
        # ----------

        # Output layer
        _p_y = T.nnet.softmax(T.dot(_z2, _W_y.T))

        return _p_y


class LSTM():

    def __init__(self, W_e, n_hidd, n_tags):

        # Dimension of the embeddings
        n_emb = W_e.shape[0]

        # MODEL PARAMETERS
        W_x = np.random.uniform(size=(4*n_hidd, n_emb))   # RNN Input layer
        W_h = np.random.uniform(size=(4*n_hidd, n_hidd))  # RNN recurrent var 
        W_c = np.random.uniform(size=(3*n_hidd, n_hidd))  # Second recurrent var 
        W_y = np.random.uniform(size=(n_tags, n_hidd))    # Output layer
        # Cast to theano GPU-compatible type
        W_e = W_e.astype(theano.config.floatX)
        W_x = W_x.astype(theano.config.floatX)
        W_h = W_h.astype(theano.config.floatX)
        W_c = W_c.astype(theano.config.floatX)
        W_y = W_y.astype(theano.config.floatX)
        # Store as shared parameters
        _W_e = theano.shared(W_e, borrow=True)
        _W_x = theano.shared(W_x, borrow=True)
        _W_h = theano.shared(W_h, borrow=True)
        _W_c = theano.shared(W_c, borrow=True)
        _W_y = theano.shared(W_y, borrow=True)

        # Class variables
        self.n_hidd = n_hidd
        self.param  = [_W_e, _W_x, _W_h, _W_c, _W_y]

    def _forward(self, _x, _h0=None, _c0=None):

        # Default initial hidden is allways set to zero
        if _h0 is None:
            h0  = np.zeros((1, self.n_hidd)).astype(theano.config.floatX)
            _h0 = theano.shared(h0, borrow=True)
        if _c0 is None:
            c0  = np.zeros((1, self.n_hidd)).astype(theano.config.floatX)
            _c0 = theano.shared(c0, borrow=True)

        # COMPUTATION GRAPH

        # Get parameters in nice form
        _W_e, _W_x, _W_h, _W_c, _W_y = self.param
        H                            = self.n_hidd

        # Embedding layer 
        _z1 = _W_e[:, _x].T

        # Per loop operation 
        def _step(_x_tm1, _h_tm1, _c_tm1, _W_x, _W_h, _W_c):
            
            # LINEAR TRANSFORMS
            # Note that all transformations per variable are stacked for
            # efficiency each individual variable is then selected using slices
            # of H size (see below)
            _z_x = T.dot(_x_tm1, _W_x.T)
            _z_h = T.dot(_h_tm1, _W_h.T) 
            _z_c = T.dot(_c_tm1, _W_c.T)

            # GATES
            # Note the subtlety: _x_tm1 and hence _z_x are flat and have size
            # (H,) _h_tm1 and _c_tm1 are not and thus have size (1, H)
            _i_t = T.nnet.sigmoid(_z_x[:H] +_z_h[:, :H] +_z_c[:, :H])
            _f_t = T.nnet.sigmoid(_z_x[H:2*H] +_z_h[:, H:2*H] +_z_c[:, H:2*H])
            _o_t = T.nnet.sigmoid(_z_x[3*H:4*H] +_z_h[:, 3*H:4*H] +_z_c[:, 2*H:3*H])
        
            # HIDDENS
            _c_t = _f_t*_c_tm1 + _i_t*T.tanh(_z_x[2*H:3*H] +_z_h[:, 2*H:3*H])    
            _h_t = _o_t*T.tanh(_c_t)
        
            return _h_t, _c_t
    
        # Unrol the loop
        _h, updates = theano.scan(_step,
                                  sequences=_z1,
                                  outputs_info=[_h0, _c0],
                                  non_sequences=[_W_x, _W_h, _W_c])
        # Just keep the first hidden, remove intermediate empty dimension
        _z2 = _h[0][:, 0, :]

        # Output layer
        _p_y = T.nnet.softmax(T.dot(_z2, _W_y.T))

        return _p_y
