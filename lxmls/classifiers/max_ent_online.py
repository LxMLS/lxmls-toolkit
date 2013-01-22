import sys
import numpy as np
import scipy.optimize.lbfgsb as opt2
from util.my_math_utils import *
import linear_classifier as lc


class MaxEnt_online(lc.LinearClassifier):
    '''Train a maxent in a online setting using stochastic gradient.

    '''
    def __init__(self,epoch_nr = 10,initial_step = 1.0, alpha = 1.0,regularizer=1.0):
        lc.LinearClassifier.__init__(self)

        # Maxent Model parameters (values for all training epochs
        # stored).
        self.params_per_epoch = []

        # Training parameters.
        self.epoch_nr = epoch_nr
        self.initial_step = initial_step
        self.alpha = alpha
        self.regularizer = regularizer
        
    def train(self,x,y):

        # Store original features to be used later.
        x_orig = x[:,:]

        # Append a column of ones to the current features.
        x = self.add_intercept_term(x)
 
        # Get dimension and number of examples of our input set.
        nr_x,nr_f = x.shape

        # Get the number of classes of our output set.
        classes = np.unique(y)
        nr_c = classes.shape[0]

        # Initialization of the classifier parameters.
        w = np.zeros((nr_f,nr_c))
	self.params_per_epoch = [] 

        # Randomize the examples.
        perm = np.random.permutation(nr_x)

        # For each training epoch.
        t = 0
        for epoch_nr in xrange(self.epoch_nr):
            # For each training example.
            objective = 0.0
            for nr in xrange(nr_x):
                t += 1
                # Set training rate for this example.
                learning_rate =  self.initial_step*np.power(t,-self.alpha)

                # Get one training example index at random.
                inst = perm[nr]

                # Get true output from training data.
                y_true = y[inst:inst+1,0]

                scores = self.get_scores(x[inst:inst+1,:],w)
                exp_scores = np.exp(scores)

                # Check for overflow.
                if(np.any(np.isinf(exp_scores))):
                    print "Overflow: removing max."
                    # In case of overflow, we remove the max.
                    max_score = np.max(scores)
                    scores -= max_score
                    exp_scores = np.exp(scores)
   
                z = exp_scores.sum()
                probs = exp_scores/z

                # Compute feature expectations.
                exp_feat = np.dot(x[inst:inst+1,:].transpose(),probs)

                # Compute empirical features for this example.
                emp_feat = np.zeros(w.shape)
                emp_feat[:,y_true] = x[inst:inst+1,:].transpose()

                # Update the model.
                objective += 0.5 * self.regularizer * l2norm_squared(w) - log(probs[0][y_true[0]])

                # Compute parameters.
                w = (1-self.regularizer*learning_rate)*w + learning_rate*(emp_feat - exp_feat)

                # Check for numerical errors.
                if(np.any(np.isnan(w))):
                    print "Error parameters became not a number."
                    return w

            objective /= nr_x

            # Test accuracy of the model in this epoch of training.
            self.trained = True           # To keep the test routine
                                          # happy.
            y_pred = self.test(x_orig,w)  # Test.
            acc = self.evaluate(y,y_pred) # Evaluation.
            print "epochs: %i objective: %f" %( epoch_nr,objective)
            print "epochs: %i accuracy: %f" %( epoch_nr,acc)
            self.trained = False          # We continue training.

        self.trained = True
        return w
