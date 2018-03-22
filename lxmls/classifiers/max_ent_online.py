import numpy as np
from lxmls.util.my_math_utils import *
import lxmls.classifiers.linear_classifier as lc


# ----------
# Train a maxent in a online setting using stochastic gradient
# ----------
class MaxEntOnline(lc.LinearClassifier):

    def __init__(self, nr_epochs=10, initial_step=1.0, alpha=1.0, regularizer=1.0, seed=1):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.nr_epochs = nr_epochs
        self.params_per_round = []
        self.initial_step = initial_step
        self.alpha = alpha
        self.regularizer = regularizer
        # use seed to generate permutation
        np.random.seed(seed)

    def train(self, x, y):
        self.params_per_round = []
        x_orig = x[:, :]
        x = self.add_intercept_term(x)
        nr_x, nr_f = x.shape
        nr_c = np.unique(y).shape[0]
        w = np.zeros((nr_f, nr_c))
        perm = np.random.permutation(nr_x)
        t = 0
        for epoch_nr in range(self.nr_epochs):
            objective = 0.0
            for nr in range(nr_x):
                t += 1
                learning_rate = self.initial_step * np.power(t, -self.alpha)
                # print learning_rate
                inst = perm[nr]
                y_true = y[inst:inst+1, 0]
                scores = self.get_scores(x[inst:inst+1, :], w)
                exp_scores = np.exp(scores)
                if np.any(np.isinf(exp_scores)):
                    print("Overflow: removing max")
                    # In case we overflow we remove the max
                    max_score = np.max(scores)
                    scores -= max_score
                    exp_scores = np.exp(scores)
                z = exp_scores.sum()
                probs = exp_scores / z
                # Compute feature expectations
                exp_feat = np.dot(x[inst:inst+1, :].transpose(), probs)
                # Compute empirical features for this example
                emp_feat = np.zeros(w.shape)
                emp_feat[:, y_true] = x[inst:inst+1, :].transpose()
                # Update the model
                objective += 0.5 * self.regularizer * \
                    l2norm_squared(w) - log(probs[0][y_true[0]])
                w = (1 - self.regularizer * learning_rate) * \
                    w + learning_rate * (emp_feat - exp_feat)
                if np.any(np.isnan(w)):
                    print("error parameters became not a number")
                    return w
            self.trained = True
            objective /= nr_x
            y_pred = self.test(x_orig, w)
            acc = self.evaluate(y, y_pred)
            self.trained = False
            print("Epochs: %i Objective: %f" % (epoch_nr, objective))
            print("Epochs: %i Accuracy: %f" % (epoch_nr, acc))
        self.trained = True
        return w
