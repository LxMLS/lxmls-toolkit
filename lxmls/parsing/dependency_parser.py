import numpy as np

from lxmls.parsing.dependency_decoder import DependencyDecoder
from lxmls.parsing.dependency_features import DependencyFeatures
from lxmls.parsing.dependency_reader import DependencyReader
from lxmls.parsing.dependency_writer import DependencyWriter


class DependencyParser:
    """
    Dependency parser class
    """

    def __init__(self):
        self.trained = False
        self.projective = False
        self.language = ""
        self.weights = []
        self.decoder = DependencyDecoder()
        self.reader = DependencyReader()
        self.writer = DependencyWriter()
        self.features = DependencyFeatures()

    def read_data(self, language):
        self.language = language
        self.reader.load(language)
        self.features.create_dictionary(self.reader.train_instances)

    def train_perceptron(self, n_epochs):
        """Trains the parser by running the averaged perceptron algorithm for n_epochs."""
        self.weights = np.zeros(self.features.n_feats)
        total = np.zeros(self.features.n_feats)
        for epoch in range(n_epochs):
            print("Epoch {0}".format(epoch+1))
            n_mistakes = 0
            n_tokens = 0
            n_instances = 0
            for instance in self.reader.train_instances:
                feats = self.features.create_features(instance)
                scores = self.features.compute_scores(feats, self.weights)
                if self.projective:
                    heads_pred = self.decoder.parse_proj(scores)
                else:
                    heads_pred = self.decoder.parse_nonproj(scores)

                for m in range(np.size(heads_pred)):
                    if heads_pred[m] != instance.heads[m]:  # mistake
                        for f in feats[instance.heads[m]][m]:
                            if f < 0:
                                continue
                            self.weights[f] += 1.0
                        for f in feats[heads_pred[m]][m]:
                            if f < 0:
                                continue
                            self.weights[f] -= 1.0
                        n_mistakes += 1
                    n_tokens += 1
                n_instances += 1
            print("Training accuracy: {0}".format(np.double(n_tokens-n_mistakes) / np.double(n_tokens)))
            total += self.weights

        self.weights = total / np.double(n_epochs)

    def train_crf_sgd(self, n_epochs, sigma, eta0=0.001):
        """Trains the parser by running the online MaxEnt algorithm for n_epochs, regularization coefficient sigma,
        and initial stepsize eta0 (which anneals as O(1/(sigma*t)))."""
        self.weights = np.zeros(self.features.n_feats)
        t = 0
        t0 = 1.0 / (sigma*eta0)
        for epoch in range(n_epochs):
            print("Epoch {0}".format(epoch+1))
            n_instances = 0
            objective = 0.0
            for instance in self.reader.train_instances:
                eta = 1.0 / (sigma * (t+t0))
                feats = self.features.create_features(instance)
                scores = self.features.compute_scores(feats, self.weights)

                # Compute marginals and log-partition function, and move away from that direction
                marginals, logZ = self.decoder.parse_marginals_nonproj(scores)
                self.weights -= eta * sigma * self.weights  # Scale the weight vector
                for h in range(np.size(marginals, 0)):
                    for m in range(1, np.size(marginals, 1)):
                        if feats[h][m] is None:
                            continue
                        for f in feats[h][m]:
                            if f < 0:
                                continue
                            self.weights[f] -= eta * marginals[h, m]

                # Compute score of the correct parse, and move the weight vector towards that direction.
                score_corr = 0.0
                for m in range(1, np.size(instance.heads)):
                    h = instance.heads[m]
                    score_corr += scores[h, m]
                    for f in feats[h][m]:
                        if f < 0:
                            continue
                        self.weights[f] += eta

                # Compute objective (w.r.t. this instance only)
                objective += 0.5*sigma*np.dot(self.weights, self.weights) - score_corr + logZ

                n_instances += 1
                t += 1

            print("Training objective: {0}".format(objective / n_instances))

    def test(self):
        n_mistakes = 0
        n_tokens = 0
        n_instances = 0
        arr_heads_pred = []
        for instance in self.reader.test_instances:
            feats = self.features.create_features(instance)
            scores = self.features.compute_scores(feats, self.weights)
            if self.projective:
                heads_pred = self.decoder.parse_proj(scores)
            else:
                heads_pred = self.decoder.parse_nonproj(scores)

            for m in range(np.size(heads_pred)):
                if heads_pred[m] != instance.heads[m]:  # mistake
                    for f in feats[instance.heads[m]][m]:
                        if f < 0:
                            continue
                    for f in feats[heads_pred[m]][m]:
                        if f < 0:
                            continue
                    n_mistakes += 1
                n_tokens += 1
            n_instances += 1
            arr_heads_pred.append(heads_pred)
        print("Test accuracy ({0} test instances): {1}".format(n_instances, np.double(n_tokens-n_mistakes) / np.double(n_tokens)))

        self.writer.save(self.language, arr_heads_pred)
