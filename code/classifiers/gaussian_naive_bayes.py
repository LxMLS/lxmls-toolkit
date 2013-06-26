import numpy as np
import scipy as scipy
import classifiers.linear_classifier as lc
from distributions.gaussian import *


class GaussianNaiveBayes(lc.LinearClassifier):
    '''Represents a Gaussian Naive Bayes classifier. 

    GaussianNaiveBayes inherits from LinearClassifier, which means
    that all operations and attributes of class LinearClassifier are
    operations and attributes of this class. This means, that we get
    all methods (to test, evaluate, ...) from the standard linear
    classifier. The only thing we need to design is a training method
    for the Naive Bayes classifier.

    '''

    def __init__(self):
        lc.LinearClassifier.__init__(self) # Calls the constructor
                                           # from LinearClassifier.
        # Naive Bayes classifier parameters.
        self.means = 0
        self.prior = 0

    def train(self,x,y):
        '''
        Trains the parameters of a Naive Bayes classifier

        '''

        # Get dimension and number of examples of our input set.
        nr_x,nr_f = x.shape

        # Get the number of classes of our output set.
        classes = np.unique(y)
        nr_c = classes.shape[0]

        # Initialization of the classifier parameters.
        prior = np.zeros(nr_c)
        means = np.zeros((nr_c,nr_f))

        # Estimation of classifier parameters (see equation 1.8) 
        # For each class in our output set.
        for i in xrange(nr_c):
            # Find all elements of input set belonging to the i^th
            # class.
            idx,_ = np.nonzero(y == classes[i])

            # Estimate i^th prior by counting number of times that the
            # i^th class appears on train set.
            prior[i] = 1.0*len(idx)/len(y)

            # Estimate i^th mean from the sample mean. But only of the
            # elements of the input set belonging to the i^th class!
            # Note: np.mean is vectorized i.e. computes the nr_x means
            #       simultaneously.
            means[i,0:2]=np.mean(x[idx,:],0)

        # Note: Remark 1.2 explains that the parameters of a Naive
        # Bayes classifier can be expressed as a linear classifier. We
        # will return the parameters in that format to be used later.
        params = np.zeros((nr_f+1,nr_c))
        for i in xrange(nr_c):
            params[0,i] = -1/2 * np.dot(means[i,:],means[i,:]) + np.log(prior[i])
            params[1:,i] = means[i].transpose()

        # Store classifier parameters.
        self.means = means
        self.prior = prior
        self.trained = True

        return params
