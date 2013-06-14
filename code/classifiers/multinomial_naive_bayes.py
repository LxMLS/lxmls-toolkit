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
        # n_docs = no. of documents
        # n_words = no. of words        
        n_docs,n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 corresponds to the fifth feature!
        
        ###########################
        # Code to be deleted
        sums = np.zeros((n_words,1))
        for i in xrange(n_classes):
            docs_in_class,_ = np.nonzero(y == classes[i]) # docs_in_class = indices of documents in class i
            prior[i] = 1.0*len(docs_in_class)/n_docs # prior = fraction of documents with this class

            word_count_in_class = x[docs_in_class,:].sum(0) # word_count_in_class = count of word occurrences in documents of class i
            sums[:,0] += word_count_in_class # sums = total number of counts of each word
            likelihood[:,i] = word_count_in_class # likelihood = count of occurrences of a word in a class
                                    # NOTE: at this point this is a count, not a likelihood
                
        for f in xrange(n_words):
            for i in xrange(n_classes):
                if self.smooth:
                    likelihood[f,i] = (self.smooth_param + likelihood[f,i])/(n_words*self.smooth_param + sums[f,0]) # Add-one smoothing
                else:
                    likelihood[f,i] = likelihood[f,i]/sums[f,0] 
        # End of code to be deleted
        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in xrange(n_classes):
            params[0,i] = np.log(prior[i])
            params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
