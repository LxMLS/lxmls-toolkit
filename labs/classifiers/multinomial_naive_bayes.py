import numpy as np
import scipy as scipy
import linear_classifier as lc
import sys
from distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self,xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 100
        
    def train(self,x,y):
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
        if 1:
            smooth_param = 1.0
            for i in xrange(nr_c):    
                idx,_ = np.nonzero(y == classes[i])
                prior[i] = 1.0*len(idx)/nr_x
                nr_features_in_c = x[idx,:].sum(0) + smooth_param
                likelihood[:,i] = nr_features_in_c/(nr_features_in_c.sum())  # 
        else:
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

        params = np.zeros((nr_f+1,nr_c))
        for i in xrange(nr_c):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
