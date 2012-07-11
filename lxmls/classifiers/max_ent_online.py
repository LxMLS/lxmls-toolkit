import sys
import numpy as np
import scipy.optimize.lbfgsb as opt2
from util.my_math_utils import *
import linear_classifier as lc

'''
Train a maxent in a online setting using stochastic gradient
'''
class MaxEnt_online(lc.LinearClassifier):

    def __init__(self,round_nr = 10,initial_step = 1.0, alpha = 1.0,regularizer=1.0):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        # Maxent Model parameters (values for all training rounds stored)
        self.params_per_round = []
        # Training parameters
        self.round_nr = round_nr
        self.initial_step = initial_step
        self.alpha = alpha
        self.regularizer = regularizer
        
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
	self.params_per_round = [] 

        ## Randomize the examples
        perm = np.random.permutation(nr_x)

        # For each training round
        for round_nr in xrange(self.round_nr):

            # Set training rate for this round
            learning_rate =  self.initial_step*np.power(t,-self.alpha)		
            # For each feature
            objective = 0.0
            for nr in xrange(nr_x):
                # Get one feature index at random
                inst = perm[nr]
                # Get true output from training data
                y_true = y[inst:inst+1,0]
                scores = self.get_scores(x[inst:inst+1,:],w)
                exp_scores = np.exp(scores)
                # Check for overflow
                if(np.any(np.isinf(exp_scores))):
                    print "overflow: removing max"
                    #in case we overflow we remove the max
                    max_score = np.max(scores)
                    scores -= max_score
                    exp_scores = np.exp(scores)                 
                z = exp_scores.sum()
                probs = exp_scores/z
                #compute feature expectations
                exp_feat = np.dot(x[inst:inst+1,:].transpose(),probs)
                #compute empirical features for this example
                emp_feat = np.zeros(w.shape)
                emp_feat[:,y_true] = x[inst:inst+1,:].transpose()
                #update the model          
                objective += 0.5 * self.regularizer * l2norm_squared(w) - log(probs[0][y_true[0]])
                # compute parameters 
                w = (1-self.regularizer*learning_rate)*w + learning_rate*(emp_feat - exp_feat)
                # check for numerical errors 
                if(np.any(np.isnan(w))):
                    print "error parameters became not a number"
                    return w

            objective /= nr_x

            # test accuracy of the model in this round of training
            # to keep the test routine happy		
            self.trained = True
            # test
            y_pred = self.test(x_orig,w)
            # evaluation
            acc = self.evaluate(y,y_pred)
            print "epochs: %i objective: %f" %( round_nr,objective)
            print "epochs: %i accuracy: %f" %( round_nr,acc)
            # We continue training
            self.trained = False

        self.trained = true
        return w
