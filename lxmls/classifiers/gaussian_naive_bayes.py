import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
from lxmls.distributions.gaussian import *


class GaussianNaiveBayes(lc.LinearClassifier):

    def __init__(self):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.means = 0
        # self.variances = 0
        self.prior = 0

    def train(self, x, y):
        nr_x, nr_f = x.shape
        nr_c = np.unique(y).shape[0]
        prior = np.zeros(nr_c)
        likelihood = np.zeros((nr_f, nr_c))
        classes = np.unique(y)
        means = np.zeros((nr_c, nr_f))
        variances = np.zeros((nr_c, nr_f))
        for i in range(nr_c):
            idx, _ = np.nonzero(y == classes[i])
            prior[i] = 1.0 * len(idx) / len(y)
            for f in range(nr_f):
                g = estimate_gaussian(x[idx, f])
                means[i, f] = g.mean
                variances[i, f] = g.variance
        # Take the mean of the covariance for each matric
        variances = np.mean(variances, 1)
        params = np.zeros((nr_f+1, nr_c))
        for i in range(nr_c):
            params[0, i] = -1/2*np.dot(means[i, :], means[i, :]) + np.log(prior[i])
            params[1:, i] = means[i].transpose()
            # params[0,i] = -1/(2*variances[i]) * np.dot(means[i,:],means[i,:]) + np.log(prior[i])
            # params[1:,i] = (1/variances[i] * means[i]).transpose()
        self.means = means
        # self.variances = variances
        self.prior = prior
        self.trained = True
        return params
