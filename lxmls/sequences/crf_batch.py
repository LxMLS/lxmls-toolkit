import sys
import numpy as np
from scipy import optimize
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb


class CRFBatch(dsc.DiscriminativeSequenceClassifier):
    """ Implements a first order CRF"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 regularizer=0.00001):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.regularizer = regularizer

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        emp_counts = self.get_empirical_counts(dataset) / len(dataset.seq_list)
        params, _, d = optimize.fmin_l_bfgs_b(self.get_objective,
                                              self.parameters,
                                              args=[dataset, emp_counts],
                                              factr=1e14,
                                              maxfun=50,
                                              iprint=2,
                                              pgtol=1e-5)
        self.parameters = params
        self.trained = True
        return params

    def get_objective(self, parameters, dataset, emp_counts):
        self.parameters = parameters
        gradient = np.zeros(parameters.shape)
        gradient += emp_counts
        objective = 0.0
        likelihoods = 0.0
        exp_counts = np.zeros(parameters.shape)
        for sequence in dataset.seq_list:
            seq_obj, seq_lik = self.get_objective_seq(parameters, sequence, exp_counts)
            objective += seq_obj
            likelihoods += seq_lik
        objective /= len(dataset.seq_list)
        likelihoods /= len(dataset.seq_list)
        exp_counts /= len(dataset.seq_list)
        objective -= 0.5 * self.regularizer * np.dot(parameters, parameters)
        objective -= likelihoods
        gradient -= self.regularizer * parameters
        gradient -= exp_counts

        # Since we are minizing we need to multiply both the objective and gradient by -1
        objective *= -1
        gradient *= -1

        if objective < 0:
            import pdb
            pdb.set_trace()

        print(objective)
        return objective, gradient

    def get_objective_seq(self, parameters, sequence, exp_counts):
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = \
            self.compute_scores(sequence)

        state_posteriors, transition_posteriors, log_likelihood = \
            self.compute_posteriors(initial_scores, transition_scores,
                                    final_scores, emission_scores)

        seq_objective = self.compute_output_score(sequence.y,
                                                  initial_scores,
                                                  transition_scores,
                                                  final_scores,
                                                  emission_scores)

        # Now compute expected counts.
        num_states = self.get_num_states()  # Number of states.
        length = len(sequence.x)  # Length of the sequence.

        for state in range(num_states):
            features = self.feature_mapper.get_initial_features(sequence, state)
            for feat_id in features:
                exp_counts[feat_id] += state_posteriors[0, state]

        for pos in range(length):
            for state in range(num_states):
                features = self.feature_mapper.get_emission_features(sequence, pos, state)
                for feat_id in features:
                    exp_counts[feat_id] += state_posteriors[pos, state]

                if pos > 0:
                    for prev_state in range(num_states):
                        features = self.feature_mapper.get_transition_features(sequence, pos-1, state, prev_state)
                        for feat_id in features:
                            exp_counts[feat_id] += transition_posteriors[pos-1, state, prev_state]

        for state in range(num_states):
            features = self.feature_mapper.get_final_features(sequence, state)
            for feat_id in features:
                exp_counts[feat_id] += state_posteriors[length-1, state]

        return seq_objective, log_likelihood

    def get_empirical_counts(self, dataset):
        """
        Computes the empirical counts for a dataset.
        Empirical counts are the counts of the features that appear in the gold data.
        """
        emp_counts = np.zeros(self.feature_mapper.get_num_features())
        for sequence in dataset.seq_list:
            y_t_true = sequence.y[0]
            true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true)
            for feat_id in true_initial_features:
                emp_counts[feat_id] += 1

            for pos in range(len(sequence.x)):
                y_t_true = sequence.y[pos]
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                for feat_id in true_emission_features:
                    emp_counts[feat_id] += 1

                if pos > 0:
                    prev_y_t_true = sequence.y[pos-1]
                    true_transition_features = self.feature_mapper.get_transition_features(
                        sequence, pos-1, y_t_true, prev_y_t_true)
                    for feat_id in true_transition_features:
                        emp_counts[feat_id] += 1

            pos = len(sequence.x)
            y_t_true = sequence.y[pos-1]
            true_final_features = self.feature_mapper.get_final_features(sequence, y_t_true)
            for feat_id in true_final_features:
                emp_counts[feat_id] += 1

        return emp_counts
