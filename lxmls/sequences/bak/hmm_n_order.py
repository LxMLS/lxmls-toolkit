import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("util/")

from my_math_utils import *
from viterbi import viterbi
from forward_backward import forward_backward, sanity_check_forward_backward


# ----------
# Sum all the entries in a dictionary of dictionaries
# ----------
def deep_dictionary_sum(dic):
    total = 0
    for key in dic.iterkeys():
        total += sum(dic[key].values())
    return total


# ---------- State Implementation
# Assuming there are y real states
# the set of HMM states z for an n order
# hmm is all combinations of y n by n
# Example:
# For y = 1,2 and n = 2
# Z = [[0,0],[0,1],[1,0],[1,1]]
# Where z_t = [y_t-1,y_t] hence
# for z = [0,1] z[0] = y_t-1 = 0
# and z[1] = y_t = 1
# ----------


# Build all possible histories
def build_state_history(model_order, nr_states):
    nr_positions = nr_states**model_order
    histories = np.zeros([nr_positions, model_order], dtype='int')
    for order in xrange(model_order):
        # print "order %i"%order
        size_of_block = nr_states**order
        number_of_iter = nr_positions / (size_of_block*nr_states)
        # print "Size of Block %i"%size_of_block
        # print "Number of iter %i"%number_of_iter
        # print "Number of positions %i"%nr_positions
        for i in xrange(number_of_iter):
            # print "Iter%i" % i
            for state in xrange(nr_states):
                # print "state%i"%state
                for bs in xrange(size_of_block):
                    # print "bs %i"%bs
                    pos = i*nr_states*size_of_block + state*size_of_block + bs
                    # print "pos %i"%(pos)
                    # print "order %i"%order
                    histories[pos, order] = state
    return histories


# Check if two histories are compatile to follow each other
# State = z_t = [y_t-1,y_t]
# Prev_state = z_t-1 = [y_t-2,y_t-1]
# if order = 2 and state = [0,1] then prev_state must start with [1,*]
# if order = 3 and state = [0,1,2] then prev_state must start with [1,2,*]
def possible_prev_state(state, prev_state):
    # if(not  state[:-1] == prev_state[1:]):
    #     print "prev"
    #     print prev_state
    #     print "state"
    #     print state
    #     print "results"
    #     print state[:-1] == prev_state[1:]
    return state[:-1] == prev_state[1:]


# NOT IMPLEMENTED YET
class HMMNOrder:

    def __init__(self, dataset, true_states=-1, order=2):
        self.trained = False

        # True states are the true number of labels
        if true_states == -1:
            self.true_states = len(dataset.int_to_pos)
        else:
            self.true_states = true_states

        # Base states are the true states +1 for the fake state ending and starting the HMM
        # Base states will be denoted by y
        self.base_states = self.true_states + 1
        # Id for the fake state for starting and ending the HMM
        self.fake_state = self.true_states

        self.nr_types = len(dataset.int_to_word)
        self.dataset = dataset
        # HMM order
        # Order = 1 means the usual first order HMM p(y_t | y_t-1)
        # Order = 2 means a second order HMM p(y_t | y_t-1, y_t-2)
        self.order = order

        self.fake_history = []
        for i in xrange(order):
            self.fake_history.append(self.fake_state)
        self.fake_history = tuple(self.fake_history)
        # HMM States - States of the trellis we will concatenat the base states for a given order to use usual forward backward
        # There will be base_state^order hmm_states
        # HMM states will be denoted by z
        self.hmm_states = self.base_states**self.order

        self.all_states = build_state_history(self.order, self.base_states).tolist()
        self.all_states = tuple(map(lambda x: tuple(x), self.all_states))
        self.fake_history_idx = -1
        for state_idx, state in enumerate(self.all_states):
            # print state,self.fake_history
            if state == self.fake_history:
                self.fake_history_idx = state_idx
        # Model counts tables
        # Transition counts - Counts of being in base_state y at time t, and
        # coming from hmm_state at time t-1 c(y_t| z_t-1)
        self.transition_counts = {}
        self.transition_probs = {}
        # Observation counts - Counts of observing word w and being in base state y at time t
        self.observation_counts = {}
        self.observation_probs = {}

    # Train a model in a supervised way, by counting events
    # Smoothing represents add-alpha smoothing
    def train_supervised(self, sequence_list, obs_smoothing=0.0, trans_smoothing=0.0):
        if len(self.dataset.int_to_pos) != self.true_states:
            print "Cannot train supervised models with number of true states different than true pos tags"
            return
        # Sets all counts to zeros
        self.clear_counts(obs_smoothing, trans_smoothing)
        self.collect_counts_from_corpus(sequence_list)
        self.update_params()

    def print_transitions(self):
        pass
        # history = range(self.order)
        # history_list = []
        # for i in xrange(self.order):
        #     for o_state in xrange(self.nr_states+1):
        #         history[i]
        #         history_list.append(history)
        # txt = "\t"
        # for hist in history_list:
        #     print "%i-%i\t"%(hist[1],hist[0])
        # for state in xrange(self.nr_states):
        #     for hist in history_list:
        #         print state
        #         print history

    def add_counts(self, strucuture, state, history, value):
        if state == 42:
            print "A entrar no add counts com 42 e valor %f" % value
            print "history"
            print history
        if history not in strucuture:
            if state == 42:
                print "history not in structure"
            strucuture[history] = {}
        if state in strucuture[history]:
            if state == 42:
                print "adding value prev: %.2f value %.2f" % (strucuture[history][state], value)
            strucuture[history][state] += value
            if state == 42:
                print "adding value after: %.2f value %.2f" % (strucuture[history][state], value)
        else:
            if state == 42:
                print "state not in history"
                print strucuture[history]
            strucuture[history][state] = value

    def set_counts(self, strucuture, state, history, value):
        if history not in strucuture:
            strucuture[history] = {}
        if state in strucuture[history]:
            strucuture[history][state] = value
        else:
            strucuture[history][state] = value

    def get_counts(self, strucuture, state, history):
        if history not in strucuture:
            return 0.0
        if state in strucuture[history]:
            return strucuture[history][state]
        else:
            return 0.0

    def collect_counts_from_corpus(self, sequence_list):
        """ Collects counts from a labeled corpus"""
        print "Collecting counts"
        history = range(self.order)
        for sequence in sequence_list.seq_list:
            len_x = len(sequence.x)
            # Goes from 0 to len(X)+order
            for pos in xrange(len_x+self.order):
                if pos >= len_x:
                    y_state = self.fake_state
                else:
                    y_state = sequence.y[pos]
                    # Only add counts in valid position
                    if sequence.x[pos] == 42:
                        print "Vou adicionar contagens"
                    self.add_counts(self.observation_counts, sequence.x[pos], y_state, 1.0)

                # Build the history for this position
                for i in xrange(1, self.order+1):
                    if pos-i < 0 or pos-i >= len_x:
                        history[-i] = self.fake_state
                    else:
                        history[-i] = sequence.y[pos-i]

                self.add_counts(self.transition_counts, y_state, tuple(history), 1.0)

    # Initializes the parameter randomnly
    def initialize_radom(self):
        pass

    def clear_counts(self, obs_smoothing=0.0, trans_smoothing=0.0):
        self.transition_counts = {}
        self.observation_counts = {}
        if obs_smoothing != 0.0:
            print "Estou no obs smoothing"
            for tt in xrange(self.true_states):
                if tt not in self.observation_counts:
                    self.observation_counts[tt] = {}
                    for w_id in xrange(self.nr_types):
                        if w_id not in self.observation_counts[tt]:
                            self.observation_counts[tt][w_id] = obs_smoothing
        if trans_smoothing != 0.0:
            print "Estou no trans smoothing"
            for hmm_s in self.all_states:
                if hmm_s not in self.transition_counts:
                    self.transition_counts[hmm_s] = {}
                for tt in xrange(self.base_states):
                    if tt not in self.transition_counts[hmm_s]:
                        self.transition_counts[hmm_s][tt] = trans_smoothing

    # Gets the counts table, normalizes it and add the values to the
    # parameters
    def update_params(self):
        self.transition_probs = self.normalize_counts(self.transition_counts)
        self.observation_probs = self.normalize_counts(self.observation_counts)

        # ----------

    # for each X in p(y|x) normalizes the values of y to sum to one.
    # Uses the values from the count table and saves the values in the params_table
    # ----------
    def normalize_counts(self, count_table):
        param_table = {}
        for condition, values in count_table.iteritems():
            total = sum(values.values())
            if total == 0:
                break
            params_values = {}
            for key, value in values.iteritems():
                params_values[key] = value / total
            param_table[condition] = params_values
        return param_table

    def update_counts(self, seq, posteriors):
        pass

    # ----------
    # Check if the collected counts make sense
    # Transition Counts  - Should sum to number of tokens - 2* number of sentences
    # Observation counts - Should sum to the number of tokens
    #
    # ----------
    def sanity_check_counts(self, seq_list):
        nr_sentences = len(seq_list.seq_list)
        nr_tokens = sum(map(lambda seq: len(seq.x), seq_list.seq_list))
        print "Nr_sentence: %i" % nr_sentences
        print "Nr_tokens: %i" % nr_tokens
        sum_transition = deep_dictionary_sum(self.transition_counts)
        sum_observations = deep_dictionary_sum(self.observation_counts)
        # Compare
        value = nr_tokens + 2*nr_sentences
        if abs(sum_transition - value) > 0.001:
            print "Transition counts do not match: is - %f should be - %f" % (sum_transition, value)
        else:
            print "Transition Counts match"
        value = nr_tokens
        if abs(sum_observations - value) > 0.001:
            print "Observations counts do not match: is - %f should be - %f" % (sum_observations, value)
        else:
            print "Observations Counts match"

    def get_seq_prob(self, seq, node_potentials, edge_potentials):
        pass

    def forward_backward(self, seq):
        # Add extra initial and end state
        N = len(seq.y) + 2
        H = self.hmm_states
        forward = np.zeros([H, N], dtype=float)
        backward = np.zeros([H, N], dtype=float)

        # Forward part
        # Initial position
        forward[self.fake_history_idx, 0] = 1
        # Middle position
        for pos in xrange(1, N-1):
            # Remove the fake position zero
            true_pos = pos - 1
            for current_state_idx, current_state in enumerate(self.all_states):
                current_y_state = current_state[-1]
                for prev_state_idx, prev_state in enumerate(self.all_states):
                    prev_y_state = prev_state[-1]
                    if possible_prev_state(current_state, prev_state):
                        forward_v = forward[prev_state_idx, pos-1]
                        if forward_v == 0:
                            continue
                        # print "Position: %i"%(pos)
                        # print "Prev State"
                        # print prev_state
                        # print "Current State"
                        # print current_state

                        trans_v = self.get_counts(self.transition_probs, current_y_state, prev_state)
                        prob = forward_v * trans_v
                        forward[current_state_idx, pos] += prob

                forward[current_state_idx, pos] *= self.get_counts(self.observation_probs,
                                                                   seq.x[true_pos],
                                                                   current_y_state)
                # print "Forward Entry %.3f"%forward[current_state_idx,pos]
        # Final Position
        for current_state_idx, current_state in enumerate(self.all_states):
            current_y_state = current_state[-1]
            if current_y_state == self.fake_state:
                # print "Final state"
                # print current_state
                for prev_state_idx, prev_state in enumerate(self.all_states):
                    if possible_prev_state(current_state, prev_state):
                        forward_v = forward[prev_state_idx, N-2]
                        if forward_v == 0:
                            continue
                        # print "Prev State"
                        # print prev_state
                        # print "Current State"
                        # print current_state
                        trans_v = self.get_counts(self.transition_probs, current_y_state, prev_state)
                        forward[current_state_idx, N-1] += forward_v * trans_v
        # Backward part
        # Final position
        for current_state_idx, current_state in enumerate(self.all_states):
            current_y_state = current_state[-1]
            if current_y_state == self.fake_state:
                backward[current_state_idx, N-1] = 1
        # Middel positions
        for pos in xrange(N-2, 0, -1):
            true_pos = pos - 1
            for current_state_idx, current_state in enumerate(self.all_states):
                current_y_state = current_state[-1]
                prob = 0
                for next_state_idx, next_state in enumerate(self.all_states):
                    next_y_state = next_state[-1]
                    if possible_prev_state(next_state, current_state):
                        # print prev_state_idx,prev_state,pos
                        back = backward[next_state_idx, pos+1]
                        trans = self.get_counts(self.transition_probs, next_y_state, current_state)
                        if true_pos+1 >= len(seq.x):
                            observation = 1
                        else:
                            observation = self.get_counts(self.observation_probs, seq.x[true_pos+1], next_y_state)
                        prob += trans * observation * back
                backward[current_state_idx, pos] = prob
        # Initial position
        prob = 0
        for next_state_idx, next_state in enumerate(self.all_states):
            next_y_state = next_state[-1]
            if possible_prev_state(next_state, self.fake_history):
                # print "next state"
                # print next_state
                back = backward[next_state_idx, 1]
                trans = self.get_counts(self.transition_probs, next_y_state, self.fake_history)
                observation = self.get_counts(self.observation_probs, seq.x[0], next_y_state)
                # print "obs %.4f"%(observation)
                # print "back %.4f"%(back)
                # print "trans %.4f"%(trans)
                # print "adding %.4f"%(trans * observation * back)
                prob += trans * observation * back
        backward[self.fake_history_idx, 0] = prob
        return forward, backward

    def sanity_check_fb(self, forward, backward):
        H, N = forward.shape
        likelihood = np.zeros([N, 1])
        for pos in xrange(N):
            aux = 0
            for current_state in xrange(H):
                aux += forward[current_state, pos] * backward[current_state, pos]
            likelihood[pos] = aux
        print likelihood
        for i in xrange(pos):
            if abs(aux - likelihood[i]) > 0.001:
                print "Likelihood for pos %i and pos %i mismatch: %f - %f" % (i, pos, likelihood[i], aux)
                return False

        return True

    # ----------
    # Returns the node posteriors
    # ----------
    def get_node_posteriors(self, seq):
        forward, backward = self.forward_backward(seq)
        # print sanity_check_forward_backward(forward,backward)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        # print likelihood
        return self.get_node_posteriors_aux(seq, forward, backward, likelihood)

    def get_node_posteriors_aux(self, seq, forward, backward, likelihood):
        H, N = forward.shape
        posteriors = np.zeros([H, N], dtype=float)
        for pos in xrange(N):
            for current_state in xrange(H):
                posteriors[current_state, pos] = forward[current_state, pos] * backward[current_state, pos] / likelihood
        return posteriors

    def get_edge_posteriors(self, seq):
        forward, backward = self.forward_backward(seq)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        return self.get_edge_posteriors_aux(seq, forward, backward, likelihood)

    def get_edge_posteriors_aux(self, seq, forward, backward, likelihood):
        H, N = forward.shape
        edge_posteriors = np.zeros([H, H, N-1], dtype=float)
        for pos in xrange(N-1):
            print "At position %i" % pos
            true_pos = pos - 1
            for prev_state_idx, prev_state in enumerate(self.all_states):
                for state_idx, state in enumerate(self.all_states):
                    state_y = state[-1]
                    if possible_prev_state(state, prev_state):
                        print "Adding edge between"
                        print "prev"
                        print prev_state
                        print "state"
                        print state
                        trans = self.get_counts(self.transition_probs, state_y, prev_state)
                        if true_pos+1 >= len(seq.y):
                            observation = 1
                        else:
                            observation = self.get_counts(self.observation_probs, seq.x[true_pos+1], state_y)
                        edge_posteriors[prev_state_idx, state_idx, pos] = forward[prev_state_idx, pos] * trans * observation * backward[state_idx, pos+1] / likelihood
                    else:
                        edge_posteriors[prev_state_idx, state_idx, pos] = 0
        return edge_posteriors

    def get_posteriors(self, seq):
        forward, backward = forward_backward(seq)
        # self.sanity_check_fb(forward,backward)
        H, N = forward.shape
        likelihood = np.sum(forward[:, N-1])
        node_posteriors = self.get_node_posteriors_aux(seq, forward, backward, likelihood)
        edge_posteriors = self.get_edge_posteriors_aux(seq, forward, backward, likelihood)
        return [node_posteriors, edge_posteriors], likelihood

    def posterior_decode(self, seq):
        posteriors = self.get_node_posteriors(seq)
        # Compute true node posteriors
        # print "Posteriors"
        # print posteriors
        res = np.argmax(posteriors, axis=0)
        # print res
        newres = np.zeros(len(seq.y), dtype='int')
        # Removed unused positions
        for i in xrange(1, res.shape[0]-1):
            newres[i-1] = self.all_states[res[i]][-1]
        new_seq = seq.copy_sequence()
        new_seq.y = newres
        # print "Res transfomed"
        # print newres
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

    # ----------
    # Plot the transition matrix for a given HMM
    # ----------
    def print_transition_matrix(self):
        cax = plt.imshow(self.transition_probs, interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        plt.xticks(np.arange(0, len(self.nr_states)), np.arange(self.nr_staets), rotation=90)
        plt.yticks(np.arange(0, len(self.nr_states)), np.arange(self.nr_staets))

    def pick_best_smoothing(self, train, test, smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
            self.train_supervised(train, smoothing=i)
            viterbi_pred_train = self.viterbi_decode_corpus(train.seq_list)
            posterior_pred_train = self.posterior_decode_corpus(train.seq_list)
            eval_viterbi_train = self.evaluate_corpus(train.seq_list, viterbi_pred_train)
            eval_posterior_train = self.evaluate_corpus(train.seq_list, posterior_pred_train)
            print "Smoothing %f --  Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (i, eval_posterior_train, eval_viterbi_train)

            viterbi_pred_test = self.viterbi_decode_corpus(test.seq_list)
            posterior_pred_test = self.posterior_decode_corpus(test.seq_list)
            eval_viterbi_test = self.evaluate_corpus(test.seq_list, viterbi_pred_test)
            eval_posterior_test = self.evaluate_corpus(test.seq_list, posterior_pred_test)
            print "Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f" % (i, eval_posterior_test, eval_viterbi_test)
            if eval_posterior_test > max_acc:
                max_acc = eval_posterior_test
                max_smooth = i
        return max_smooth
