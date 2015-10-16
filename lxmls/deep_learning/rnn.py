#!/usr/bin/python

import numpy as np
import theano
import theano.tensor as T

from ipdb import set_trace

class RNN():

    def __init__(self, n_words, n_hidd, n_tags):

        # MODEL PARAMETERS
        W_x = np.random.uniform(size=(n_hidd, n_words)) # Input layer 
        W_h = np.random.uniform(size=(n_hidd, n_hidd))  # Recurrent layer
        W_y = np.random.uniform(size=(n_tags, n_hidd))  # Output layer
        # Cast to theano GPU-compatible type
        W_x = W_x.astype(theano.config.floatX)
        W_h = W_h.astype(theano.config.floatX)
        W_y = W_y.astype(theano.config.floatX)
        # Store as shared parameters
        _W_x = theano.shared(W_x, borrow=True)
        _W_h = theano.shared(W_h, borrow=True)
        _W_y = theano.shared(W_y, borrow=True)

        # Class variables
        self.n_hidd = n_hidd
        self.param  = [_W_x, _W_h, _W_y]

    def _forward(self, _x, _h0=None):

        # Default initial hidden is allways set to zero
        if _h0 is None:
            h0  = np.zeros((1, self.n_hidd)).astype(theano.config.floatX)
            _h0 = theano.shared(h0, borrow=True)

        # COMPUTATION GRAPH
        _W_x, _W_h, _W_y = self.param

        # NOTE: Since _x contains the indices rather than full one-hot vectors,
        # we use  _W_x[:, x_tm1].T instead of T.dot(x_tm1, _W_x.T)

        ###########################
        # Solution to Exercise 6.3 
    
        # This defines what to do at each step
        def rnn_step(x_tm1, h_tm1, _W_x, W_h):
            return T.nnet.sigmoid(_W_x[:, x_tm1].T + T.dot(h_tm1, W_h.T))
    
        # This creates the variable length computation graph (unrols the rnn)
        _h, updates = theano.scan(fn=rnn_step, 
                                  sequences=_x, 
                                  outputs_info=dict(initial=_h0),
                                  non_sequences=[_W_x ,_W_h])
    
        # Remove intermediate empty dimension
        _z2 = _h[:,0,:]
    
        # End of solution to Exercise 6.3
        ###########################

        # Output layer
        _p_y = T.nnet.softmax(T.dot(_z2, _W_y.T))

        return _p_y


def lstm(_x, _h0, _c0, _W_x, _W_h, _W_c, H):

    '''
    LSTM Recurrent Neural Network. 

    x, h0, c0      theano.tensor.matrix of shapes [L, I], [1, H], [1, H]
                   x is the sequence input to the network, h0 and c0 initial
                   values for the hidden variables.
                   L = number of samples, I = number of features, H = size of
                   hidden 
    W_x, W_h, W_c  theano.shared variables of shapes [4*H, I], [4*H, H], [4*H, H]
                   These are the weights of the LSTM. Note that all
                   transformations per variable are stacked for efficiency

    '''

    # Per loop operation 
    def _step(_x_tm1, _h_tm1, _c_tm1, _W_x, _W_h, _W_c):
        
        # LINEAR TRANSFORMS
        # Note that all transformations per variable are stacked for efficiency
        # each individual variable is then selected using slices of H size
        # (see below)
        _z_x = T.dot(_x_tm1, _W_x.T)
        _z_h = T.dot(_h_tm1, _W_h.T)
        _z_c = T.dot(_c_tm1, _W_c.T)
    
        # GATES
        # Note the subtlety: _x_tm1 and hence _z_x are flat and have size (H,),
        # _h_tm1 and _c_tm1 are not and thus have size (1, H)
        _i_t = T.nnet.sigmoid(_z_x[:H] +_z_h[:, :H] +_z_c[:, :H])
        _f_t = T.nnet.sigmoid(_z_x[H:2*H] +_z_h[:, H:2*H] +_z_c[:, H:2*H])
        _o_t = T.nnet.sigmoid(_z_x[3*H:4*H] +_z_h[:, 3*H:4*H] +_z_c[:, 2*H:3*H])
    
        # HIDDENS
        _c_t = _f_t*_c_tm1 + _i_t*T.tanh(_z_x[2*H:3*H] +_z_h[:, 2*H:3*H])    
        _h_t = _o_t*T.tanh(_c_t)
    
        return _h_t, _c_t

    # Unrol the loop
    _h, updates = theano.scan(_step,
                              sequences=_x,
                              outputs_info=[_h0, _c0],
                              non_sequences=[_W_x, _W_h, _W_c])
    # Just keep the first hidden, remove intermediate empty dimension
    _h = _h[0][:, 0, :]

    return _h, updates
