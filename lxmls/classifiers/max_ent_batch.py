import sys
import numpy as np
import scipy.optimize.lbfgsb as opt2
from lxmls.util.my_math_utils import *
import lxmls.classifiers.linear_classifier as lc


# ----------
# Train a maxent in a batch setting using LBFGS
# ----------
class MaxEntBatch(lc.LinearClassifier):

    def __init__(self, regularizer=1):
        self.parameters = 0
        self.regularizer = regularizer

    def train(self, x, y):
        x = self.add_intercept_term(x)
        nr_x, nr_f = x.shape
        classes = np.unique(y)
        nr_c = classes.shape[0]
        # Add the bias feature
        init_parameters = np.zeros((nr_f, nr_c), dtype=float)
        emp_counts = np.zeros((nr_f, nr_c))
        classes_idx = []
        for c_i, c in enumerate(classes):
            idx, _ = np.nonzero(y == c)
            classes_idx.append(idx)
            emp_counts[:, c_i] = x[idx, :].sum(0)
        params = self.minimize_lbfgs(init_parameters, x, y, self.regularizer, emp_counts, classes_idx, nr_x, nr_f, nr_c)
        self.trained = True
        return params

    def minimize_lbfgs(self, parameters, x, y, sigma, emp_counts, classes_idx, nr_x, nr_f, nr_c):
        parameters2 = parameters.reshape([nr_f*nr_c], order="F")
        result, _, d = opt2.fmin_l_bfgs_b(self.get_objective, parameters2, args=[x, y, sigma, emp_counts, classes_idx, nr_x, nr_f, nr_c])
        return result.reshape([nr_f, nr_c], order="F")

    # ----------
    # Obj =  -sum_(x,y) p(y|x) + sigma*||w||_2^2
    # Obj = \sum_(x,y) -w*f(x,y) + log(\sum_(y') exp(w*f(x,y'))) +  sigma*||w||_2^2
    # ----------
    def get_objective(self, parameters, x, y, sigma, emp_counts, classes_idx, nr_x, nr_f, nr_c):
        parameters2 = parameters.reshape([nr_f, nr_c], order="F")
        # f(x,y).w
        #       scores = spdot(x,parameters2)
        scores = np.dot(x, parameters2)
        # exp(f(x,y).w)
        exp_scores = np.exp(scores)
        # sum_y exp(f(x,y).w)
        z = exp_scores.sum(axis=1).reshape([nr_x, 1], order="F")
        # log sum_y exp(f(x,y).w)
        logz = np.log(z)
        sum_scores = 0
        for i, classes in enumerate(classes_idx):
            sum_scores += np.sum(scores[classes, i])

        #
        objective = -sum_scores/nr_x + np.sum(logz)/nr_x + 0.5*sigma*l2norm_squared(parameters2)
        # Probs
        probs = exp_scores / z
        # exp_feat = spdot(x.transpose(),probs)
        exp_feat = np.dot(x.transpose(), probs)
        grad = exp_feat/nr_x + parameters2*sigma - emp_counts/nr_x
        print("Objective = {0}".format(objective))
        return objective, grad.reshape([nr_f*nr_c], order="F")
