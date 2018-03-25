import numpy as np
import lxmls.sequences.sequence_classifier as sc
import lxmls.sequences.confusion_matrix as cm
from lxmls.sequences.log_domain import *
import pdb


class HMM(sc.SequenceClassifier):
    """ Implements a first order HMM."""

    def __init__(self, observation_labels, state_labels):
        """Initialize an HMM. observation_labels and state_labels are the sets
        of observations and states, respectively. They are both LabelDictionary
        objects."""
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)

        num_states = self.get_num_states()
        num_observations = self.get_num_observations()

        # Vector of probabilities for the initial states: P(state|START).
        self.initial_probs = np.zeros(num_states)

        # Matrix of transition probabilities: P(state|previous_state).
        # First index is the state, second index is the previous_state.
        self.transition_probs = np.zeros([num_states, num_states])

        # Vector of probabilities for the final states: P(STOP|state).
        self.final_probs = np.zeros(num_states)

        # Matrix of emission probabilities. Entry (k,j) is probability
        # of observation k given state j.
        self.emission_probs = np.zeros([num_observations, num_states])

        # Count tables.
        self.initial_counts = np.zeros(num_states)
        self.transition_counts = np.zeros([num_states, num_states])
        self.final_counts = np.zeros(num_states)
        self.emission_counts = np.zeros([num_observations, num_states])

    def train_EM(self, dataset, smoothing=0, num_epochs=10, evaluate=True, seed=1):
        self.initialize_random(seed)

        if evaluate:
            acc = self.evaluate_EM(dataset)
            print("Initial accuracy: %f" % acc)

        for t in range(1, num_epochs):
            # E-Step
            total_log_likelihood = 0.0
            self.clear_counts(smoothing)
            for sequence in dataset.seq_list:
                # Compute scores given the observation sequence.
                initial_scores, transition_scores, final_scores, emission_scores = \
                    self.compute_scores(sequence)

                state_posteriors, transition_posteriors, log_likelihood = \
                    self.compute_posteriors(initial_scores,
                                            transition_scores,
                                            final_scores,
                                            emission_scores)
                self.update_counts(sequence, state_posteriors, transition_posteriors)
                total_log_likelihood += log_likelihood

            print("Iter: %i Log Likelihood: %f" % (t, total_log_likelihood))
            # M-Step
            self.compute_parameters()
            if evaluate:
                # Evaluate accuracy at this iteration
                acc = self.evaluate_EM(dataset)
                print("Iter: %i Accuracy: %f" % (t, acc))

    def evaluate_EM(self, dataset):
        # Evaluate accuracy at initial iteration
        pred = self.viterbi_decode_corpus(dataset)
        confusion_matrix = cm.build_confusion_matrix(dataset.seq_list, pred,
                                                     self.get_num_states(), self.get_num_states())
        best = cm.get_best_assignment(confusion_matrix)
        new_pred = []
        for i, sequence in enumerate(dataset.seq_list):
            pred_seq = pred[i]
            new_seq = pred_seq.copy_sequence()
            for j, y_hat in enumerate(new_seq.y):
                new_seq.y[j] = best[y_hat]
            new_pred.append(new_seq)
        acc = self.evaluate_corpus(dataset, new_pred)
        return acc

    def train_supervised(self, dataset, smoothing=0):
        """ Train an HMM from a list of sequences containing observations
        and the gold states. This is just counting and normalizing."""
        # Set all counts to zeros (optionally, smooth).
        self.clear_counts(smoothing)
        # Count occurrences of events.
        self.collect_counts_from_corpus(dataset)
        # Normalize to get probabilities.
        self.compute_parameters()

    def collect_counts_from_corpus(self, dataset):
        """ Collects counts from a labeled corpus."""
        for sequence in dataset.seq_list:
            # Take care of first position.
            self.initial_counts[sequence.y[0]] += 1
            self.emission_counts[sequence.x[0], sequence.y[0]] += 1

            # Take care of intermediate positions.
            for i, x in enumerate(sequence.x[1:]):
                y = sequence.y[i+1]
                y_prev = sequence.y[i]
                self.emission_counts[x, y] += 1
                self.transition_counts[y, y_prev] += 1

            # Take care of last position.
            self.final_counts[sequence.y[-1]] += 1

    # Initializes the parameter randomnly
    def initialize_random(self, seed=1):
        np.random.seed(seed)
        jitter = 1
        num_states = self.get_num_states()
        num_observations = self.get_num_observations()

        self.initial_counts.fill(1)
        self.initial_counts += jitter * np.random.rand(num_states)
        self.transition_counts.fill(1)
        self.transition_counts += jitter * np.random.rand(num_states, num_states)
        self.emission_counts.fill(1)
        self.emission_counts += jitter * np.random.rand(num_observations, num_states)
        self.final_counts.fill(1)
        self.final_counts += jitter * np.random.rand(num_states)
        self.compute_parameters()
        self.clear_counts()

    def clear_counts(self, smoothing=0):
        """ Clear all the count tables."""
        self.initial_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.final_counts.fill(smoothing)
        self.emission_counts.fill(smoothing)

    def update_counts(self, sequence, state_posteriors, transition_posteriors):
        """ Used in the E-step in EM."""

        # ----------
        # Solution to Exercise 2.10

        num_states = self.get_num_states()  # Number of states.
        length = len(sequence.x)  # Length of the sequence.

        # Take care of initial probs
        for y in range(num_states):
            self.initial_counts[y] += state_posteriors[0, y]
        for pos in range(length):
            x = sequence.x[pos]
            for y in range(num_states):
                self.emission_counts[x, y] += state_posteriors[pos, y]
                if pos > 0:
                    for y_prev in range(num_states):
                        self.transition_counts[y, y_prev] += transition_posteriors[pos-1, y, y_prev]

        # Final position
        for y in range(num_states):
            self.final_counts[y] += state_posteriors[length-1, y]

            # End of solution to Exercise 2.10
            # ----------

    def compute_parameters(self):
        """ Estimate the HMM parameters by normalizing the counts."""

        # Normalize the initial counts.
        self.initial_probs = self.initial_counts / np.sum(self.initial_counts)

        # Normalize transition counts
        self.transition_probs = self.transition_counts / (np.sum(self.transition_counts, 0)+self.final_counts)

        # Normalize final counts
        self.final_probs = self.final_counts / (np.sum(self.transition_counts, 0)+self.final_counts)

        # Normalize emission counts
        self.emission_probs = self.emission_counts / np.sum(self.emission_counts, 0)

    def compute_scores(self, sequence):
        length = len(sequence.x)  # Length of the sequence.
        num_states = self.get_num_states()  # Number of states of the HMM.

        # Initial position.
        initial_scores = np.log(self.initial_probs)

        # Intermediate position.
        emission_scores = np.zeros([length, num_states]) + logzero()
        transition_scores = np.zeros([length-1, num_states, num_states]) + logzero()
        for pos in range(length):
            emission_scores[pos, :] = np.log(self.emission_probs[sequence.x[pos], :])
            if pos > 0:
                transition_scores[pos-1, :, :] = np.log(self.transition_probs)

        # Final position.
        final_scores = np.log(self.final_probs)

        return initial_scores, transition_scores, final_scores, emission_scores

    # ----------
    # Plot the transition matrix for a given HMM
    # ----------
    def print_transition_matrix(self):
        import matplotlib.pyplot as plt
        cax = plt.imshow(self.transition_probs, interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        plt.xticks(np.arange(0, self.get_num_states()), self.state_labels.names, rotation=90)
        plt.yticks(np.arange(0, self.get_num_states()), self.state_labels.names)
        plt.show()

    def pick_best_smoothing(self, train, test, smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
            self.train_supervised(train, smoothing=i)
            viterbi_pred_train = self.viterbi_decode_corpus(train)
            posterior_pred_train = self.posterior_decode_corpus(train)
            eval_viterbi_train = self.evaluate_corpus(train, viterbi_pred_train)
            eval_posterior_train = self.evaluate_corpus(train, posterior_pred_train)
            print("Smoothing %f --  Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (i, eval_posterior_train, eval_viterbi_train))

            viterbi_pred_test = self.viterbi_decode_corpus(test)
            posterior_pred_test = self.posterior_decode_corpus(test)
            eval_viterbi_test = self.evaluate_corpus(test, viterbi_pred_test)
            eval_posterior_test = self.evaluate_corpus(test, posterior_pred_test)
            print("Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (i, eval_posterior_test, eval_viterbi_test))
            if eval_posterior_test > max_acc:
                max_acc = eval_posterior_test
                max_smooth = i
        return max_smooth
