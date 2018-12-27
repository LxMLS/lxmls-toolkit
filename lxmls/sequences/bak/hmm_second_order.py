import sys
import numpy as np

import matplotlib.pyplot as plt

sys.path.append("util/")

from my_math_utils import *
from viterbi import viterbi
from forward_backward import forward_backward, sanity_check_forward_backward


class HMMSecondOrder:
    """ Implements an HMM with second order dependecies
    We will use the state expansion version where we expand the states to be n^2 and use the normal infernece algorithms (forward backward and viterbi)"""

    def __init__(self, dataset, nr_states=-1):
        self.trained = False
        self.init_probs = 0
        self.transition_probs = 0
        self.observation_probs = 0
        if nr_states == -1:
            self.nr_states = len(dataset.int_to_pos)
        else:
            self.nr_states = nr_states
        self.nr_types = len(dataset.int_to_word)
        self.dataset = dataset
        # Model order
        self.order = 2
        # Model counts tables
        # c(s_t = s | s_t-1 = q, s_t-2=s)
        # Should increase the number of states by one to include the faque state at begining and end
        self.transition_counts = np.zeros([self.nr_states+1, self.nr_states+1, self.nr_states+1], dtype=float)
        # c(o_t = v | s_t = s)
        # Increase the number of states to account for the faque state
        self.observation_counts = np.zeros([self.nr_types+1, self.nr_states+1], dtype=float)

    def get_number_states(self):
        self.nr_states

    # Train a model in a supervised way, by counting events
    # Smoothing represents add-alpha smoothing
    def train_supervised(self, sequence_list, smoothing=0):
        if len(self.dataset.int_to_pos) != self.nr_states:
            print "Cannot train supervised models with number of states different than true pos tags"
            return

        nr_types = len(sequence_list.x_dict)
        nr_states = len(sequence_list.y_dict)
        # Sets all counts to zeros
        self.clear_counts(smoothing)
        self.collect_counts_from_corpus(sequence_list)
        self.update_params()

    def print_transitions(self, table):
        txt = "\t"
        for prev_state in xrange(self.nr_states+1):
            for prev_prev_state in xrange(self.nr_states+1):
                txt += "%i-%i\t" % (prev_state, prev_prev_state)
        print txt

        for state in xrange(self.nr_states+1):
            txt = "%i\t" % state
            for prev_state in xrange(self.nr_states+1):
                for prev_prev_state in xrange(self.nr_states+1):
                    txt += "%.3f\t" % (table[state, prev_state, prev_prev_state])
            print txt

    def print_observations(self, table):
        txt = "\t"
        for obs in xrange(self.nr_types+1):
            txt += "%i\t" % obs
        print txt

        for state in xrange(self.nr_states+1):
            txt = "%i\t" % state
            for obs in xrange(self.nr_types+1):
                txt += "%.3f\t" % (table[obs, state])
            print txt

    def collect_counts_from_corpus(self, sequence_list):
        """ Collects counts from a labeled corpus"""
        for sequence in sequence_list.seq_list:
            len_x = len(sequence.x)
            # Goes from 0 to len(X)+order
            for pos in xrange(len_x+self.order):
                if pos >= len_x:
                    y_state = self.nr_states
                    x_idx = self.nr_types
                else:
                    y_state = sequence.y[pos]
                    x_idx = sequence.x[pos]
                self.observation_counts[x_idx, y_state] += 1

                # Take care of prev_prev_state
                if pos-2 < 0:
                    prev_prev_state = self.nr_states
                else:
                    prev_prev_state = sequence.y[pos-2]
                # Take care of prev_state
                if pos-1 < 0:
                    prev_state = self.nr_states
                elif pos-1 >= len_x:
                    prev_state = self.nr_states
                else:
                    prev_state = sequence.y[pos-1]
                self.transition_counts[y_state, prev_state, prev_prev_state] += 1

    # Initializes the parameter randomnly
    def initialize_radom(self):
        jitter = 1
        self.transition_counts.fill(1)
        self.transition_counts += jitter * np.random.rand(
            self.transition_counts.shape[0],
            self.transition_counts.shape[1],
            self.transition_counts.shape[2])
        self.observation_counts.fill(1)
        self.observation_counts += jitter * np.random.rand(
            self.observation_counts.shape[0],
            self.observation_counts.shape[1])
        self.update_params()
        self.clear_counts()

    def clear_counts(self, smoothing=0):
        self.transition_counts.fill(smoothing)
        self.observation_counts.fill(smoothing)

    def update_params(self):
        # Normalize
        # Update transitions
        self.transition_probs = np.zeros(self.transition_counts.shape)
        for prev_state in xrange(self.nr_states+1):
            for prev_prev_state in xrange(self.nr_states+1):
                sum_value = 0
                for state in xrange(self.nr_states+1):
                    sum_value += self.transition_counts[state, prev_state, prev_prev_state]
                for state in xrange(self.nr_states+1):
                    if sum_value != 0:
                        self.transition_probs[
                            state, prev_state, prev_prev_state] = self.transition_counts[
                            state, prev_state, prev_prev_state] / sum_value
        self.observation_probs = normalize_array(self.observation_counts)

    def update_counts(self, seq, posteriors):
        pass
        # node_posteriors,edge_posteriors = posteriors
        # #print"Node Posteriors"
        # #print node_posteriors
        # H,N = node_posteriors.shape
        # ## Take care of initial probs
        # #print "seq_x"
        # #print seq.x
        # for y in xrange(H):
        #     self.init_counts[y] += node_posteriors[y,0]
        #     x = seq.x[0]
        #     #print "x_%i=%i"%(0,x)
        #     self.observation_counts[x,y] += node_posteriors[y,0]
        # for pos in xrange(1,N-1):
        #     x = seq.x[pos]
        #     #print "x_%i=%i"%(pos,x)
        #     for y in xrange(H):
        #         self.observation_counts[x,y] += node_posteriors[y,pos]
        #         for y_next in xrange(H):
        #             ## pos-1 since edge_posteriors are indexed by prev_edge and not current edge
        #             self.transition_counts[y_next,y] += edge_posteriors[y,y_next,pos-1]

        # ##Final position
        # for y in xrange(H):
        #     x = seq.x[N-1]
        #     #print "x_%i=%i"%(N-1,x)
        #     self.observation_counts[x,y] += node_posteriors[y,N-1]
        #     for y_next in xrange(H):
        #         self.final_counts[y_next,y] += edge_posteriors[y,y_next,N-2]

    # ----------
    # Check if the collected counts make sense
    # Transition Counts  - Should sum to number of tokens - 2* number of sentences
    # Observation counts - Should sum to the number of tokens
    #
    # Seq_list should be the same used for train.
    # NOTE: If you use smoothing when trainig you have to account for that when comparing the values
    # ----------
    def sanity_check_counts(self, seq_list, smoothing=0):
        nr_sentences = len(seq_list.seq_list)
        nr_tokens = sum(map(lambda seq: len(seq.x), seq_list.seq_list))
        print "Nr_sentence: %i" % nr_sentences
        print "Nr_tokens: %i" % nr_tokens
        sum_transition = np.sum(self.transition_counts)
        sum_observations = np.sum(self.observation_counts)
        # Compare
        value = (nr_tokens + 2*nr_sentences) + smoothing*self.transition_counts.size
        if abs(sum_transition - value) > 0.001:
            print "Transition counts do not match: is - %f should be - %f" % (sum_transition, value)
        else:
            print "Transition Counts match"
        value = nr_tokens + 2*nr_sentences + self.observation_counts.size*smoothing
        if abs(sum_observations - value) > 0.001:
            print "Observations counts do not match: is - %f should be - %f" % (sum_observations, value)
        else:
            print "Observations Counts match"

    # ----------
    # Edge Potentials:
    # edge_potentials(y_t=a,y_t-1=b,yt-2=a,t)
    # Node potentials:
    # node_potentials(y_t = a, t)
    # ----------
    def build_potentials(self, sequence):
        nr_states = self.nr_states
        nr_pos = len(sequence.x)
        edge_potentials = np.zeros([self.nr_states+1, self.nr_states+1, self.nr_states+1, nr_pos+2], dtype=float)
        node_potentials = np.zeros([self.nr_states+1, nr_pos+2], dtype=float)
        for pos in xrange(nr_pos+2):
            edge_potentials[:, :, :, pos] = self.transition_probs
            if pos >= nr_pos:
                node_potentials[:, pos] = self.observation_probs[self.nr_types, :]
            else:
                node_potentials[:, pos] = self.observation_probs[sequence.x[pos], :]
        return node_potentials, edge_potentials

    # ----------
    # Gets the probability of a given sequence
    # ----------
    def get_seq_prob(self, seq, node_potentials, edge_potentials):
        nr_pos = len(seq.x)
        value = 1
        for pos in xrange(nr_pos+2):
            if pos >= nr_pos:
                y_state = self.nr_states
            else:
                y_state = seq.y[pos]
            if pos-1 >= nr_pos:
                prev_y_state = self.nr_states
            elif pos-1 < 0:
                prev_y_state = self.nr_states
            else:
                prev_y_state = seq.y[pos-1]
            if pos-2 < 0:
                prev_prev_y_state = self.nr_states
            else:
                prev_prev_y_state = seq.y[pos-2]
            value *= node_potentials[y_state, pos] * edge_potentials[y_state, prev_y_state, prev_prev_y_state, pos]
        return value

    def forward_backward(self, seq):
        node_potentials, edge_potentials = self.build_potentials(seq)
        forward, backward = forward_backward(node_potentials, edge_potentials)
        sanity_check_forward_backward(forward, backward)
        return forward, backward

    def sanity_check_fb(self, forward, backward):
        return sanity_check_forward_backward(forward, backward)

    # ----------
    # Returns the node posterios
    # ----------
    def get_node_posteriors(self, seq):
        nr_states = self.nr_states
        node_potentials, edge_potentials = self.build_potentials(seq)
        forward, backward = forward_backward(node_potentials, edge_potentials)
        # print sanity_check_forward_backward(forward,backward)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        # print likelihood
        return self.get_node_posteriors_aux(seq, forward, backward, node_potentials, edge_potentials, likelihood)

    def get_node_posteriors_aux(self, seq, forward, backward, node_potentials, edge_potentials, likelihood):
        H, N = forward.shape
        posteriors = np.zeros([H, N], dtype=float)

        for pos in xrange(N):
            for current_state in xrange(H):
                posteriors[current_state, pos] = forward[current_state, pos] * backward[current_state, pos] / likelihood
        return posteriors

    def get_edge_posteriors(self, seq):
        nr_states = self.nr_states
        node_potentials, edge_potentials = self.build_potentials(seq)
        forward, backward = forward_backward(node_potentials, edge_potentials)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        return self.get_edge_posteriors_aux(seq, forward, backward, node_potentials, edge_potentials, likelihood)

    def get_edge_posteriors_aux(self, seq, forward, backward, node_potentials, edge_potentials, likelihood):
        H, N = forward.shape
        edge_posteriors = np.zeros([H, H, N-1], dtype=float)
        for pos in xrange(N-1):
            for prev_state in xrange(H):
                for state in xrange(H):
                    edge_posteriors[prev_state, state, pos] = forward[prev_state, pos] * edge_potentials[prev_state, state, pos] * node_potentials[
                        state, pos+1] * backward[state, pos+1] / likelihood
        return edge_posteriors

    def get_posteriors(self, seq):
        nr_states = self.nr_states
        node_potentials, edge_potentials = self.build_potentials(seq)
        forward, backward = forward_backward(node_potentials, edge_potentials)
        # self.sanity_check_fb(forward,backward)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        node_posteriors = self.get_node_posteriors_aux(seq, forward, backward, node_potentials, edge_potentials, likelihood)
        edge_posteriors = self.get_edge_posteriors_aux(seq, forward, backward, node_potentials, edge_potentials, likelihood)
        return [node_posteriors, edge_posteriors], likelihood

    def posterior_decode(self, seq):
        posteriors = self.get_node_posteriors(seq)
        res = np.argmax(posteriors, axis=0)
        new_seq = seq.copy_sequence()
        new_seq.y = res
        return new_seq

    def posterior_decode_corpus(self, seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.posterior_decode(seq))
        return predictions

    def viterbi_decode(self, seq):
        node_potentials, edge_potentials = self.build_potentials(seq)
        viterbi_path, _ = viterbi(node_potentials, edge_potentials)
        res = viterbi_path
        new_seq = seq.copy_sequence()
        new_seq.y = res
        return new_seq

    def viterbi_decode_corpus(self, seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.viterbi_decode(seq))
        return predictions

    def evaluate_corpus(self, seq_list, predictions):
        total = 0.0
        correct = 0.0
        for i, seq in enumerate(seq_list):
            pred = predictions[i]
            for i, y_hat in enumerate(pred.y):
                if seq.y[i] == y_hat:
                    correct += 1
                total += 1
        return correct / total
