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
        nr_x,nr_f = x.shape
        # nr_x = no. of documents
        # nr_f = no. of words
        nr_c = np.unique(y).shape[0]
        # nr_c = no. of classes
        prior = np.zeros(nr_c)
#        ind_per_class = {}
        classes = np.unique(y)
#        for i in xrange(nr_c):
#            idx,_ = np.nonzero(y == classes[i])
#            ind_per_class = idx
        likelihood = np.zeros((nr_f,nr_c))
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

        params = np.zeros((nr_f+1,nr_c))
        for i in xrange(nr_c):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
