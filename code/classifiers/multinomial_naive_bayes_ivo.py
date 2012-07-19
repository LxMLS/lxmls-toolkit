import numpy as np
import scipy as scipy
import linear_classifier as lc

class MultinomialNaiveBayes(lc.LinearClassifier):

        def __init__(self):
                lc.LinearClassifier.__init__(self)
                self.trained = False

        def train(self,x,y):

                classes = np.unique(y)
                nr_classes = len(classes)
                nr_instances, nr_features = x.shape

                params = np.zeros((nr_features, nr_classes))

                for i in xrange(nr_classes):
                        idx = np.nonzero(y == classes[i])
                        prior_i = float(len(idx)) / float(nr_instances)
                        prior_i = len(idx) / nr_instances
                        for f in xrange(nr_features):
                                #likelihood_if = np.sum(x[idx,f]) / np.sum(x[:,f])
                                likelihood_if = (1 + np.sum(x[idx,f])) / (np.sum(x[:,f]) + nr_features)

                                params[f,i] = prior_i * likelihood_if

                self.trained = True

                return params