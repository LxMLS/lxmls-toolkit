from __future__ import division
import numpy as np
from scipy.misc import logsumexp
import pickle  # To store classes on files
import theano
import theano.tensor as T


def index2onehot(index, N):
    """
    Transforms index to one-hot representation, for example

    Input: e.g. index = [1, 2, 0], N = 4
    Output:     [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    """
    L = index.shape[0]
    onehot = np.zeros((N, L))
    for l in np.arange(L):
        onehot[index[l], l] = 1
    return onehot


class NumpyMLP:
    """
    Basic MLP with forward-pass and gradient computation
    """

    def __init__(self, geometry, actvfunc, rng=None, model_file=None):
        """
        Input: geometry  tuple with sizes of layer

        Input: actvfunc  list of strings indicating the type of activation
                         function. Supported 'sigmoid', 'softmax'

        Input: rng       string inidcating random seed
        """

        # Generate random seed if not provided
        if rng is None:
            rng = np.random.RandomState(1234)

        # CHECK THE PARAMETERS ARE VALID
        self.sanity_checks(geometry, actvfunc)

        # THIS DEFINES THE MLP
        self.n_layers = len(geometry) - 1
        if model_file:
            if geometry or actvfunc:
                raise ValueError("If you load a model geometry and actvfunc"
                                 "should be None")
            self.params, self.actvfunc = self.load(model_file)
        else:
            # Parameters are stored as [weight0, bias0, weight1, bias1, ... ]
            # for consistency with the theano way of storing parameters
            self.params = self.init_weights(rng, geometry, actvfunc)
            self.actvfunc = actvfunc

    def forward(self, x, all_outputs=False):
        """
        Forward pass

        all_outputs = True  return intermediate activations
        """

        # This will store activations at each layer and the input. This is
        # needed to compute backpropagation
        if all_outputs:
            activations = []

            # Input
        tilde_z = x

        for n in range(self.n_layers):

            # Get weigths and bias of the layer (even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # Linear transformation
            z = np.dot(W, tilde_z) + b

            # Non-linear transformation
            if self.actvfunc[n] == "sigmoid":
                tilde_z = 1.0 / (1+np.exp(-z))

            elif self.actvfunc[n] == "softmax":
                # Softmax is computed in log-domain to prevent
                # underflow/overflow
                tilde_z = np.exp(z - logsumexp(z, 0))

            if all_outputs:
                activations.append(tilde_z)

        if all_outputs:
            tilde_z = activations

        return tilde_z

    def grads(self, x, y):
        """
       Computes the gradients of the network with respect to cross entropy
       error cost
       """

        # Run forward and store activations for each layer
        activations = self.forward(x, all_outputs=True)

        # For each layer in reverse store the gradients for each parameter
        nabla_params = [None] * (2*self.n_layers)

        for n in np.arange(self.n_layers-1, -1, -1):

            # Get weigths and bias (always in even and odd positions)
            # Note that sometimes we need the weight from the next layer
            W = self.params[2*n]
            b = self.params[2*n+1]
            if n != self.n_layers-1:
                W_next = self.params[2*(n+1)]

            # ----------
            # Solution to Exercise 6.2

            # If it is the last layer, compute the average cost gradient
            # Otherwise, propagate the error backwards from the next layer
            if n == self.n_layers-1:
                # NOTE: This assumes cross entropy cost
                if self.actvfunc[n] == 'sigmoid':
                    e = (activations[n]-y) / y.shape[0]
                elif self.actvfunc[n] == 'softmax':
                    I = index2onehot(y, W.shape[0])
                    e = (activations[n]-I) / y.shape[0]

            else:
                e = np.dot(W_next.T, e)
                # This is correct but confusing n+1 is n in the guide
                e *= activations[n] * (1-activations[n])

            # Weight gradient
            nabla_W = np.zeros(W.shape)
            for l in np.arange(e.shape[1]):
                if n == 0:
                    # For the first layer, the activation is the input
                    nabla_W += np.outer(e[:, l], x[:, l])
                else:
                    nabla_W += np.outer(e[:, l], activations[n-1][:, l])
            # Bias gradient
            nabla_b = np.sum(e, 1, keepdims=True)

            # End of solution to Exercise 6.2
            # ----------

            # Store the gradients
            nabla_params[2*n] = nabla_W
            nabla_params[2*n+1] = nabla_b

        return nabla_params

    def init_weights(self, rng, geometry, actvfunc):
        """
       Following theano tutorial at
       http://deeplearning.net/software/theano/tutorial/
       """
        params = []
        for n in range(self.n_layers):
            n_in, n_out = geometry[n:n+2]
            weight = rng.uniform(low=-np.sqrt(6./(n_in+n_out)),
                                 high=np.sqrt(6./(n_in+n_out)),
                                 size=(n_out, n_in))
            if actvfunc[n] == 'sigmoid':
                weight *= 4
            elif actvfunc[n] == 'softmax':
                weight *= 4
            bias = np.zeros((n_out, 1))
            # Append parameters
            params.append(weight)
            params.append(bias)
        return params

    def sanity_checks(self, geometry, actvfunc):

        # CHECK ACTIVATIONS
        if actvfunc:
            # Supported actvfunc
            supported_acts = ['sigmoid', 'softmax']
            if geometry and (len(actvfunc) != len(geometry)-1):
                raise ValueError("The number of layers and actvfunc does not match")
            elif any([act not in supported_acts for act in actvfunc]):
                raise ValueError("Only these actvfunc supported %s" % (" ".join(supported_acts)))
            # All internal layers must be a sigmoid
            for internal_act in actvfunc[:-1]:
                if internal_act != 'sigmoid':
                    raise ValueError("Intermediate layers must be sigmoid")

    def save(self, model_path):
        """
        Save model
        """
        par = self.params + self.actvfunc
        with open(model_path, 'wb') as fid:
            pickle.dump(par, fid, pickle.HIGHEST_PROTOCOL)

    def load(self, model_path):
        """
        Load model
        """
        with open(model_path) as fid:
            par = pickle.load(fid, pickle.HIGHEST_PROTOCOL)
            params = par[:len(par)//2]
            actvfunc = par[len(par)//2:]
        return params, actvfunc

    def plot_weights(self, show=True, aspect='auto'):
        """
       Plots the weights of the newtwork
       """
        import matplotlib.pyplot as plt
        plt.figure()
        for n in range(self.n_layers):
            # Get weights
            W = self.params[2*n]
            b = self.params[2*n+1]

            plt.subplot(2, self.n_layers, n+1)
            plt.imshow(W, aspect=aspect, interpolation='nearest')
            plt.title('Layer %d Weight' % n)
            plt.colorbar()
            plt.subplot(2, self.n_layers, self.n_layers+(n+1))
            plt.plot(b)
            plt.title('Layer %d Bias' % n)
            plt.colorbar()
        if show:
            plt.show()


class TheanoMLP(NumpyMLP):
    """
    MLP VERSION USING THEANO
    """

    def __init__(self, geometry, actvfunc, rng=None, model_file=None):
        """
        Input: geometry  tuple with sizes of layer

        Input: actvfunc  list of strings indicating the type of activation
                         function. Supported 'sigmoid', 'softmax'

        Input: rng       string inidcating random seed
        """

        # Generate random seed if not provided
        if rng is None:
            rng = np.random.RandomState(1234)

        # This will call NumpyMLP.__init__.py intializing
        # Defining: self.n_layers self.params self.actvfunc
        NumpyMLP.__init__(self, geometry, actvfunc, rng=rng, model_file=model_file)

        # The parameters in the Theano MLP are stored as shared, borrowed
        # variables. This data will be moved to the GPU when used
        # use self.params.get_value() and self.params.set_value() to acces or
        # modify the data in the shared variables
        self.shared_params()

        # Symbolic variables representing the input and reference output
        x = T.matrix('x')
        y = T.ivector('y')  # Index of the correct class (int32)

        # Compile forward function
        self.fwd = theano.function([x], self._forward(x))
        # Compile list of gradient functions
        self.grs = [theano.function([x, y], _gr) for _gr in self._grads(x, y)]

    def forward(self, x):
        # Ensure the type matches theano selected type
        x = x.astype(theano.config.floatX)
        return self.fwd(x)

    def grads(self, x, y):
        # Ensure the type matches theano selected type
        x = x.astype(theano.config.floatX)
        y = y.astype('int32')
        return [gr(x, y) for gr in self.grs]

    def shared_params(self):

        params = [None] * (2*self.n_layers)
        for n in range(self.n_layers):
            # Get Numpy weigths and bias (always in even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # IMPORTANT: Ensure the types in the variables and theano operations
            # match. This is ofte a source of errors
            W = W.astype(theano.config.floatX)
            b = b.astype(theano.config.floatX)

            # Store as shared variables
            # Note that, unlike numpy, broadcasting is not active by default
            W = theano.shared(value=W, borrow=True)
            b = theano.shared(value=b, borrow=True, broadcastable=(False, True))

            # Keep in mind that naming variables is useful when debugging
            W.name = 'W%d' % (n+1)
            b.name = 'b%d' % (n+1)

            # Store weight and bias, now as theano shared variables
            params[2*n] = W
            params[2*n+1] = b

        # Overwrite our params
        self.params = params

    def _forward(self, x, all_outputs=False):
        """
        Symbolic forward pass

        all_outputs = True  return symbolic input and intermediate activations
        """

        # This will store activations at each layer and the input. This is
        # needed to compute backpropagation
        if all_outputs:
            activations = [x]

            # Input
        tilde_z = x

        # ----------
        # Solution to Exercise 6.4
        for n in range(self.n_layers):

            # Get weigths and bias (always in even and odd positions)
            W = self.params[2*n]
            b = self.params[2*n+1]

            # Linear transformation
            z = T.dot(W, tilde_z) + b

            # Keep in mind that naming variables is useful when debugging
            # see e.g. theano.printing.debugprint(tilde_z)
            z.name = 'z%d' % (n+1)

            # Non-linear transformation
            if self.actvfunc[n] == "sigmoid":
                tilde_z = T.nnet.sigmoid(z)
            elif self.actvfunc[n] == "softmax":
                tilde_z = T.nnet.softmax(z.T).T

            # Name variable
            tilde_z.name = 'tilde_z%d' % (n+1)

            if all_outputs:
                activations.append(tilde_z)
        # End of solution to Exercise 6.4
        # ----------

        if all_outputs:
            tilde_z = activations

        return tilde_z

    def _cost(self, x, y):
        """
        Symbolic average negative log-likelihood using the soft-max output
        """
        p_y = self._forward(x)
        return -T.mean(T.log(p_y)[y, T.arange(y.shape[0])])

    def _grads(self, x, y):
        """
        Symbolic gradients
        """
        # Symbolic gradients with respect to the cost
        return [T.grad(self._cost(x, y), param) for param in self.params]
