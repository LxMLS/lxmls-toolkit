import numpy as np
import scipy as scipy
import classifiers.linear_classifier as lc
from distributions.gaussian import *

class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        # Multinomial Naive Bayes parameters
        self.means = 0
        self.prior = 0
        # Smoothing (see Exercise 1.2, point 3)
        self.smooth = True
        self.smooth_param = 100

    '''
    Trains the parameters of a Multinomial Naive Bayes classifier
    '''
    def train(self,x,y):
        # Get dimension and number of examples (word present or not 1/0) of our input set
        nr_x,nr_f = x.shape
        # Get the number of classes (topics) of our output set
        classes = np.unique(y)
        nr_c = classes.shape[0]

        # Initialization of classifier parameters
        prior = np.zeros(nr_c)
        class_count = np.zeros((nr_f,nr_c)) 
        total_count = np.zeros((nr_f,1))

        # Estimation of classifier parameters (see Equation 1.10) 
        # For each class (topic) in our output set
        for i in xrange(nr_c):
            # Find all elements of input set belonging to the i^th class
            idx,_ = np.nonzero(y == classes[i])
            # Estimate i^th prior by counting number of times that the i^th class appears on train set
            prior[i] = 1.0*len(idx)/len(y)
            # compute word counts for this class
            class_count[:,i] = sum(x[idx,:],0)
            total_count[:,0] += sum(x[idx,:],0)

	# Compute likelihood from counts, special
        if self.smooth:
            likelihood = (self.smooth_param + class_count)/(total_count + nr_f*self.smooth_param) 	

        else: 
            likelihood = class_count/total_count

        # Note: Remark 1.2 explains that the parameters of a Naive Bayes classifier can be expressed as a linear classifier. We will return the parameters in that format to be used later.
        params = np.zeros((nr_f+1,nr_c))
        for i in xrange(nr_c):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))

        # Store classifier parameters anyway
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True

        return params
