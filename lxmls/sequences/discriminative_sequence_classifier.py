import numpy as np
import lxmls.sequences.sequence_classifier as sc
import pdb


class DiscriminativeSequenceClassifier(sc.SequenceClassifier):

    def __init__(self, observation_labels, state_labels, feature_mapper):
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)

        # Set feature mapper and initialize parameters.
        self.feature_mapper = feature_mapper
        self.parameters = np.zeros(self.feature_mapper.get_num_features())

    # ----------
    #  Build the node and edge potentials
    # node - f(t,y_t,X)*w
    # edge - f(t,y_t,y_(t-1),X)*w
    # Only supports binary features representation
    # If we have an HMM with 4 positions and transitins
    # a - b - c - d
    # the edge potentials have at position:
    # 0 a - b
    # 1 b - c
    # ----------
    def compute_scores(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros(num_states)
        transition_scores = np.zeros([length-1, num_states, num_states])
        final_scores = np.zeros(num_states)

        # Initial position.
        for tag_id in range(num_states):
            initial_features = self.feature_mapper.get_initial_features(sequence, tag_id)
            score = 0.0
            for feat_id in initial_features:
                score += self.parameters[feat_id]
            initial_scores[tag_id] = score

        # Intermediate position.
        for pos in range(length):
            for tag_id in range(num_states):
                emission_features = self.feature_mapper.get_emission_features(sequence, pos, tag_id)
                score = 0.0
                for feat_id in emission_features:
                    score += self.parameters[feat_id]
                emission_scores[pos, tag_id] = score
            if pos > 0:
                for tag_id in range(num_states):
                    for prev_tag_id in range(num_states):
                        transition_features = self.feature_mapper.get_transition_features(
                            sequence, pos, tag_id, prev_tag_id)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[pos-1, tag_id, prev_tag_id] = score

        # Final position.
        for prev_tag_id in range(num_states):
            final_features = self.feature_mapper.get_final_features(sequence, prev_tag_id)
            score = 0.0
            for feat_id in final_features:
                score += self.parameters[feat_id]
            final_scores[prev_tag_id] = score

        return initial_scores, transition_scores, final_scores, emission_scores
