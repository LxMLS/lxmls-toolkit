import numpy as np
import scipy as scipy
import classifiers.linear_classifier as lc
from distributions.gaussian import *
<<<<<<< HEAD:lxmls/classifiers/multinomial_naive_bayes.py
=======

>>>>>>> upstream/master:code/classifiers/multinomial_naive_bayes.py

class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self):
        lc.LinearClassifier.__init__(self)

        # Multinomial Naive Bayes parameters.
        self.means = 0
        self.prior = 0

        # Smoothing (see Exercise 1.2, point 3).
        self.smooth = True
        self.smooth_param = 100

    def train(self,x,y):
<<<<<<< HEAD:lxmls/classifiers/multinomial_naive_bayes.py
        '''Trains the parameters of a Multinomial Naive Bayes classifier.

        '''

        # Get dimension and number of examples (word present
        # or not 1/0) of our input set.
        nr_x,nr_f = x.shape

        # Get the number of classes (topics) of our output set.
        classes = np.unique(y)
        nr_c = classes.shape[0]

        # Initialization of classifier parameters.
        prior = np.zeros(nr_c)
        class_count = np.zeros((nr_f,nr_c)) 
        total_count = np.zeros((nr_f,1))

        # Estimation of classifier parameters (see Equation 1.10).
        # For each class (topic) in our output set.
        for i in xrange(nr_c):
            # Find all elements of input set belonging to the i^th
            # class.
            idx,_ = np.nonzero(y == classes[i])

            # Estimate i^th prior by counting number of times that the
            # i^th class appears on train set.
            prior[i] = 1.0*len(idx)/len(y)

            # Compute word counts for this class.
            class_count[:,i] = sum(x[idx,:],0)
            total_count[:,0] += sum(x[idx,:],0)
=======
        # nr_x = no. of documents
        # nr_f = no. of words        
        nr_x,nr_f = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        # nr_c = no. of classes
        nr_c = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(nr_c)
        likelihood = np.zeros((nr_f,nr_c))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 corresponds to the fifth feature!
        
        ###########################
        # Code to be deleted
        sums = np.zeros((nr_f,1))
        for i in xrange(nr_c):
            idx,_ = np.nonzero(y == classes[i]) 
            prior[i] = 1.0*len(idx)/len(y) # prior = fraction of documents with this class

            value = x[idx,:].sum(0)
            sums[:,0] += value # sums = total number of counts of each word
            likelihood[:,i] = value # likelihood = count of occurrences of a word in a class
                                    # NOTE: at this point this is a count, not a likelihood
                
        for f in xrange(nr_f):
            for i in xrange(nr_c):
                if self.smooth:
                    likelihood[f,i] = (self.smooth_param + likelihood[f,i])/(nr_f*self.smooth_param + sums[f,0]) # Add-one smoothing
                else:
                    likelihood[f,i] = likelihood[f,i]/sums[f,0] 
        # End of code to be deleted
        ###########################
>>>>>>> upstream/master:code/classifiers/multinomial_naive_bayes.py

	# Compute likelihood from counts, special????????
        if self.smooth:
            likelihood = (self.smooth_param + class_count)/(total_count + nr_f*self.smooth_param) 	

        else: 
            likelihood = class_count/total_count

        # Note: Remark 1.2 explains that the parameters of a Naive
        #       Bayes classifier can be expressed as a linear
        #       classifier. We will return the parameters in that
        #       format to be used later.
        params = np.zeros((nr_f+1,nr_c))
        for i in xrange(nr_c):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))

        # Store the classifier parameters.
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True

        return params
