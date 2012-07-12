import sys
import numpy as np
from classifiers import linear_classifier as lc
from util.my_math_utils import *

class Mira(lc.LinearClassifier):

    def __init__(self,nr_epochs = 10,regularizer = 1.0, averaged = True):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        # MIRA Model parameters (values for all training epochs stored)
        self.params_per_epoch = []
        # Training parameters
        self.nr_epochs = nr_epochs
        self.regularizer = regularizer
        self.averaged = averaged

    '''
    Trains the parameters of MIRA
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
        for epoch_nr in xrange(self.nr_epochs):
            # For each training example
            for nr in xrange(nr_x):
                # Get one training example index at random
                inst = perm[nr]
 		# Get true output from training data
                y_true = y[inst:inst+1,0]
                # Predict output using the current model
                y_hat = self.get_label(x[inst:inst+1,:],w)
		## Compute loss 	
                scores = self.get_scores(x[inst:inst+1,:],w)
                true_margin = scores[:,y_true]
                predicted_margin = scores[:,y_hat]
                dist = np.abs(y_true-y_hat)
                loss = predicted_margin - true_margin  +  dist
                # If there is a prediction error
                if(y_hat != y_true):
                    ## Compute stepsize
                    if( predicted_margin == true_margin):
                        stepsize = 1/self.regularizer
                    else:
                        #stepsize = np.min([1/self.agress,loss/l2norm_squared(true_margin-predicted_margin)])
                        stepsize = np.min([1/self.regularizer,loss/l2norm_squared(x[inst:inst+1])])
                    # Add feature with respect to the true output
                    w[:,y_true] += stepsize*x[inst:inst+1,:].transpose()
                    # Subtract feature with respect to the predicted output
                    w[:,y_hat] -= stepsize*x[inst:inst+1,:].transpose()

            # Store the parameters for this epoch
            self.params_per_epoch.append(w.copy())   
            # Test accuracy of the model in this epoch of training
            # To keep the test routine happy
            self.trained = True
            # Test
            y_pred = self.test(x_orig,w)
            # Evaluation
            acc = self.evaluate(y,y_pred)
            print "Rounds: %i Accuracy: %f" %( epoch_nr,acc)
            # We continue training
            self.trained = False


        # Optionaly return the average of all training epoch parameters
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            w=new_w.copy()

        self.trained = True
        
        return w
