import sys
import numpy as np
import linear_classifier as lc
from util.my_math_utils import *

class SVM(lc.LinearClassifier):

    def __init__(self,nr_epochs = 10, initial_step = 1.0, alpha = 1.0,regularizer = 1.0):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        # SVN Model parameters (values for all training epochs stored)
        self.params_per_epoch = []
        # Training parameters
        self.nr_epochs = nr_epochs
        self.regularizer = regularizer
        self.initial_step = initial_step
        self.alpha = alpha
        
    '''
    Trains the parameters of a SVN
    '''
    def train(self,x,y):

        # Store original features to be used later
        x_orig = x[:,:]
        # Append a column of ones to the current features
        x = self.add_intercept_term(x)

        # Get dimension and number of examples of our input set
        nr_x,nr_f = x.shape
        # Get the number of classes of our output set
        classes = np.unique(y)
        nr_c = classes.shape[0]

	# Initialization of classifier parameters
        w = np.zeros((nr_f,nr_c))
	self.params_per_epoch = []

        ## Randomize the examples
        perm = np.random.permutation(nr_x)

        # For each training epoch
        t=0
        for epoch_nr in xrange(self.nr_epochs):
            # For each training example
            objective = 0.0
            for nr in xrange(nr_x):
                t += 1
                # Set training rate for this example
                learning_rate =  self.initial_step*np.power(t,-self.alpha)           
                # Get one training example index at random 
                inst = perm[nr]
                # Get true output from training data
                y_true = y[inst:inst+1,0]
                # Compute cost augmented predicion 
                scores = self.get_scores(x[inst:inst+1,:],w)
                cost_augmented_loss = scores + 1 
                cost_augmented_loss[:,y_true] -= 1 
                y_hat = np.argmax(cost_augmented_loss,axis=1).transpose()
                # Update model 
                objective += 0.5 * self.regularizer * l2norm_squared(w) - scores[:,y_true] + cost_augmented_loss[:,y_hat]
                # compute parameters
                w = (1-self.regularizer*learning_rate)*w
                w[:,y_true] += learning_rate*x[inst:inst+1,:].transpose()
                w[:,y_hat] -= learning_rate*x[inst:inst+1,:].transpose()

            objective /= nr_x

            # Store the parameters for this epoch 
            self.params_per_epoch.append(w.copy())   
            # test accuracy of the model in this epoch of training
            # to keep the test routine happy		
            self.trained = True
            # test
            y_pred = self.test(x_orig,w)
            # evaluation
            acc = self.evaluate(y,y_pred)
            print "Epochs: %i Objective: %f" %( epoch_nr,objective)
            print "Epochs: %i Accuracy: %f" %( epoch_nr,acc)
            # We continue training
            self.trained = False

        self.trained = True
        return w

