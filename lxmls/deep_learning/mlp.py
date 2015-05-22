import numpy as np
from scipy.misc import logsumexp
import cPickle  # To store classes on files
import theano
import theano.tensor as T

def index2onehot(index, N):
    '''
    Transforms index to one-hot representation, for example

    Input: e.g. index = [1, 2, 0], N = 4
    Output:     [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    '''
    L      = index.shape[0]
    onehot = np.zeros((N, L))
    for l in np.arange(L):
        onehot[index[l], l] = 1 
    return onehot    

def load_mlp(model_path):
    '''
    Load a MLP file
    '''
    with open(model_path, 'rb') as fid: 
        new_model = cPickle.load(fid)
    return new_model

class MLP():
    '''
    Basic MLP with forward-pass and gradient computation
    '''
    def __init__(self, geometry=None, weights=None, actvfunc=None, rng=None):
        '''
        Input: geometry  tuple with sizes of layer
        Input: weights   list of lists containing each weight, bias pair for
                         each layer
        Input: actvfunc  list of strings indicating the type of activation 
                         function 
        Input: rng       Random seed
        '''  

        # CHECK THE PARAMETERS ARE IN THE RIGHT FORMAT
        self.sanity_checks(geometry, weights, actvfunc)

        # INITIALIZE ACCORDING TO GIVEN PARAMETERS
        # If no actvfunc given use sigmoid with last layer being softmax
        if not actvfunc:
            if geometry:
                actvfunc = ['sigmoid']*(len(geometry)-2) + ['softmax']
            else: 
                actvfunc = ['sigmoid']*(len(weights)-1)  + ['softmax']
        # If only geometry is given, initialize weights from it with a 
        # random seed
        if geometry:         
            if not rng: 
                rng = np.random.RandomState(1234)
            weights = self.random_weigth_init(rng, geometry, actvfunc)
   
        # THIS DEFINES THE MLP  
        self.weights  = weights   
        self.actvfunc = actvfunc   
        self.n_layers = len(weights)
      
    def forward(self, x, backProp=False):
        '''
        Forward pass, apply each layer after the other
        ''' 
        # This will store activations at each layer if needed. 
        if backProp:
            activations = [x]  
        for n, W, activ in zip(np.arange(self.n_layers), self.weights, 
                               self.actvfunc):
            # Linear transformation 
            z = np.dot(W[0], x) + W[1]
            # Non-linear transformation 
            if activ == "sigmoid":
                x = 1.0/(1+np.exp(-z)) 
            elif activ == "softmax": 
                x = np.exp(z - logsumexp(z, 0))
            if backProp:
                activations.append(x)
        if backProp:
            return activations 
        else:
            return x

    def backprop_grad(self, x, y):
       '''
       Computes the gradients of the network with respect to cross entropy 
       error cost
       '''
       # Run forward and store activations for each layer 
       activations = self.forward(x, backProp=True) 
       # For each layer in reverse store the gradients we compute
       delta_weights = [None]*self.n_layers 
       for n in np.arange(len(self.weights)-1, -1, -1):
           # If it is the last layer, compute the cost gradient
           # Note: This assumes softmax, see sanity_checks()
           if n == self.n_layers-1:
               I  = index2onehot(y, self.weights[-1][0].shape[0])
               e  = activations[n+1] - I 
           # Otherwise, propagate the error one step backwards
           # Note: This assumes sigmoid, see sanity_checks()
           else:
               e  = np.dot(self.weights[n+1][0].T, e)
               e *= np.multiply(activations[n+1], (1-activations[n+1]))
           # Compute the weight gradient from the errors
           delta_W = np.zeros(self.weights[n][0].shape)
           for l in np.arange(e.shape[1]):
              delta_W += np.outer(e[:, l], activations[n][:, l])
           # Bias gradient
           delta_w = np.sum(e, 1, keepdims=True)
           # Store this gradients 
           delta_weights[n] = [delta_W, delta_w]
       return delta_weights 

    def random_weigth_init(self, rng, geometry, actvfunc):
       '''
       Following theano tutorial at 
       http://deeplearning.net/software/theano/tutorial/
       ''' 
       weights = []
       for n in np.arange(len(geometry)-1):
           n_in, n_out = geometry[n:n+2] 
           layer_weights = np.asarray(rng.uniform(
                   low=-np.sqrt(6. / (n_in + n_out)),
                   high=np.sqrt(6. / (n_in + n_out)),
                   size=(n_out, n_in)))
           if actvfunc[n] == 'sigmoid':
               layer_weights *= 4
           elif actvfunc[n] == 'softmax':
               layer_weights *= 4
           layer_bias = np.zeros((n_out, 1))
           weights.append([layer_weights, layer_bias])
       return weights    
 
    def sanity_checks(self, geometry, weights, actvfunc):
 
        # CHECK GEOMETRY OR WEIGHTS
        if (not geometry and not weights) or (geometry and weights):
            raise ValueError, "Either geometry or weights have to be defined"
        if weights and rng:   
            raise ValueError, ("It makes no sense to provide both a random"
                               "seed and the initialized layer weights")
    
        # CHECK ACTIVATIONS
        if actvfunc:
            # Forgot list
            if not isinstance(actvfunc, list):
                raise ValueError, ("actvfunc must be a *list* of strings "
                                   "indicating activations")
            # Supported actvfunc
            supported_acts = ['sigmoid', 'softmax']
            if geometry and (len(actvfunc) != len(geometry)-1):
                raise ValueError, "The number of layers and actvfunc does not match"
            elif weights and (len(actvfunc) != len(weights)):
                raise ValueError, "The number of layers and actvfunc does not match"
            elif any([act not in supported_acts for act in actvfunc]):   
                raise ValueError, ("Only these actvfunc supported %s" 
                                   % (" ".join(supported_acts)))
            # Last layer must be a softmax
            if actvfunc[-1] != 'softmax': 
                raise ValueError, "Last layer must be a softmax"
            # All internal layers must be a sigmoid
            for act in actvfunc[:-1]:
                if act != 'sigmoid':
                    raise ValueError, "Intermediate layers must be sigmoid"
 
    def save(self, model_path):
        '''
        Save model
        '''
        with open(model_path, 'wb') as fid: 
            cPickle.dump(self, fid, cPickle.HIGHEST_PROTOCOL)

    def plot_weights(self, show=True, aspect='auto'):
       '''
       Plots the weights of the newtwork
       '''
       import matplotlib.pyplot as plt
       plt.figure()
       for n, w in enumerate(self.weights):
           plt.subplot(2, self.n_layers, n+1)
           plt.imshow(w[0], aspect=aspect, interpolation='nearest') 
           plt.title('Layer %d Weight' % n)
           plt.colorbar()
           plt.subplot(2, self.n_layers, self.n_layers + (n+1))
           plt.plot(w[1])
           plt.title('Layer %d Bias' % n)
           plt.colorbar()
       if show:
           plt.show() 



class TheanoMLP(MLP):
    '''
    MLP VERSION USING THEANO
    '''
    def __init__(self, geometry=None, weights=None, actvfunc=None, rng=None):

        # Initialize MLP as ususal
        MLP.__init__(self, geometry=geometry, weights=weights, 
                     actvfunc=actvfunc, rng=rng)

        # Compile forward pass and cost gradients
        self.compile_forward() 
        self.compile_grads()

    def forward(self, x):
        return self.th_forward(x)

    def backprop_grad(self, x, y):
        return [[gr[0](x,y), gr[1](x,y)]  for gr in self.grad_comp] 

    def compile_forward(self):
        '''
        Compile the forward pass as a theano function
        '''
        # These will store the outputs at each layer including the initial 
        # input and the weights respectively
        self.tilde_z = [T.matrix('x')]
        self.params  = []
        for W, activ in zip(self.weights, self.actvfunc):
            # Turn weights into theano shared vars 
            _W = theano.shared(value=W[0], borrow=True)
            _b = theano.shared(value=W[1], borrow=True, broadcastable=(False, True))
            # Linear transformation
            z = T.dot(_W, self.tilde_z[-1]) + _b
            # Non-linear transformation 
            if activ == "sigmoid":
                self.tilde_z.append(T.nnet.sigmoid(z))
            elif activ == "softmax": 
                self.tilde_z.append(T.nnet.softmax(z.T).T)
            self.params.append([_W, _b])
        # Get a function returning the forward pass
        self.th_forward = theano.function([self.tilde_z[0]], self.tilde_z[-1]) 

    def compile_grads(self):
        '''
        Compile the gradients of training cost as a theano function
        '''       
        # Labels considered as indicator vectors
        self.y = T.ivector('y')         
        # Symbolic average negative log-likelihood using the soft-max output
        self.F = -T.sum(T.log(self.tilde_z[-1].T)[T.arange(self.y.shape[0]), 
                                                  self.y]) 
        # Compute gradients   
        self.grads     = [] 
        self.grad_comp = []
        for param in self.params:   
            # Symbolic gradients for that layer 
            self.grads.append([T.grad(self.F, param[0]),
                               T.grad(self.F, param[1])])
            # Function returning gradient for that layer
            self.grad_comp.append([theano.function([self.tilde_z[0], self.y], 
                                                   self.grads[-1][0]), 
                                   theano.function([self.tilde_z[0], self.y], 
                                                   self.grads[-1][1])])

    def compile_train(self, train_set, batch_size=400, lrate=0.01):
        '''
        Same as compile grads, but also includes the update rule
        '''       
        # Get sizes and check for coherence 
        L = train_set[0].shape[1]   
        n_batch = L/batch_size         
        if L < batch_size:
            raise ValueError, ("Batch size %d too large for %d train examples"
                               % (batch_size, L)) 
        # Store for later use
        self.n_batch = n_batch 
        # Convert the train and devel sets into a shared dataset
        train_set2=[[], []]

        train_set2[0] = theano.shared(np.asarray(train_set[0],
                                                 dtype=theano.config.floatX),
                                      borrow=True)
        train_set2[1] = T.cast(theano.shared(np.asarray(train_set[1],
                                             dtype=theano.config.floatX),
                                             borrow=True), 'int32')
        index = T.lscalar() # Symbolic index to a batch

        # Pack the update rules for ach parameters into a list
        updates = []
        for n, param, grad in zip(np.arange(self.n_layers), self.params, self.grads):
            updates.append((param[0], param[0] - lrate * grad[0]))
            updates.append((param[1], param[1] - lrate * grad[1]))

        self.train_batch = theano.function(inputs=[index], outputs=self.F,
            updates=updates,
            givens={
                self.tilde_z[0]: train_set2[0][:, index*batch_size:(index + 1)*batch_size],
                self.y: train_set2[1][index * batch_size:(index + 1) * batch_size]})
