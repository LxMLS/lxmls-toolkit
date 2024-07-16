
import sys
import numpy as np
import lxmls.classifiers.linear_classifier as lc
from lxmls.util.my_math_utils import *


class Mira(lc.LinearClassifier):

    def __init__(self, nr_rounds=10, regularizer=1.0, averaged=True):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.nr_rounds = nr_rounds
        self.regularizer = regularizer
        self.params_per_round = []
        self.averaged = averaged

    def train(self, x, y, seed=1):
        self.params_per_round = []
        x_orig = x[:, :]
        x = self.add_intercept_term(x)
        nr_x, nr_f = x.shape
        nr_c = np.unique(y).shape[0]
        w = np.zeros((nr_f, nr_c))
        for round_nr in range(self.nr_rounds):
            for nr in range(nr_x):

                # use seed to generate permutation
                np.random.seed(seed)
                # generate a permutation on length nr_x
                perm = np.random.permutation(nr_x)

                # change the seed so next epoch we don't get the same permutation
                seed += 1

                inst = perm[nr]
                scores = self.get_scores(x[inst:inst+1, :], w)
                y_true = y[inst:inst+1, 0]
                y_hat = self.get_label(x[inst:inst+1, :], w)

                true_margin = scores[:, y_true]
                predicted_margin = scores[:, y_hat]
                dist = np.abs(y_true - y_hat)
                # # Compute loss
                loss = predicted_margin - true_margin + dist
                # Compute stepsize
                if y_hat != y_true:
                    if predicted_margin == true_margin:
                        stepsize = 1 / self.regularizer
                    else:
                        # stepsize = np.min([1/self.agress,loss/l2norm_squared(true_margin-predicted_margin)])
                        stepsize = np.min([1/self.regularizer, loss[0,0]/l2norm_squared(x[inst:inst+1])])
                    w[:, y_true] += stepsize * x[inst:inst+1, :].transpose()
                    w[:, y_hat] -= stepsize * x[inst:inst+1, :].transpose()

            self.params_per_round.append(w.copy())
            self.trained = True
            y_pred = self.test(x_orig, w)
            acc = self.evaluate(y, y_pred)
            self.trained = False
            print("Rounds: %i Accuracy: %f" % (round_nr, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_round:
                new_w += old_w
            new_w /= len(self.params_per_round)
            return new_w
        return w
