import numpy as np
from lxmls.sequences.log_domain import *
import pdb
import numpy as np
from lxmls.sequences.log_domain import *
import pdb
class SequenceClassificationDecoder():
    ''' Implements a sequence classification decoder.'''

    def __init__(self):
        pass

    def run_forward(self, in_scores, trans_scores, final_scores, em_scores):
        """
        Computes the forward trellis for a given sequence.
        Receives:

        - in_scores:
        - trans_scores:
        - final_scores:
        - em_scores:
        :return:
        """
        # Length of the sequence.
        length = np.size(em_scores, 0)

        # Number of states.
        num_states = np.size(in_scores)

        # Forward variables.
        forward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        forward[0,:] = em_scores[0,:] + in_scores

        # Forward loop.
        for pos in xrange(1,length):
            for current_state in xrange(num_states):
                # Note the fact that multiplication in log domain turns a sum and sum turns a logsum
                logv = forward[pos-1, :] + trans_scores[pos-1, current_state, :]
                forward[pos, current_state] = logsum(logv) +  em_scores[pos, current_state]

        # Termination.
        log_likelihood = logsum(forward[length-1,:] + final_scores)

        return log_likelihood, forward


    def run_backward(self, in_scores, trans_scores, final_scores, em_scores):
        """
        Computes the backward trellis for a given sequence.

        Receives:
        :param in_scores:   (num_states) array
        :param trans_scores: (length-1, num_states, num_states) array
        :param final_scores: (num_states) array
        :param em_scores: (length, num_states) array

        :return:
         - log_likelihood
         - backward
        """
        # Length of the sequence.
        length = np.size(em_scores, 0)

        # Number of states.
        num_states = np.size(in_scores)

        # Backward variables.
        backward = np.zeros([length, num_states]) + logzero()

        # Initialization.
        backward[length-1,:] = final_scores

        # Backward loop.
        for pos in xrange(length-2,-1,-1):
            for current_state in xrange(num_states):
                backward[pos, current_state] = logsum(backward[pos+1, :] + trans_scores[pos, :, current_state] + em_scores[pos+1, :])

        # Termination.
        log_likelihood = logsum(backward[0,:] + in_scores + em_scores[0,:])

        return log_likelihood, backward


    def run_viterbi(self, in_scores, trans_scores, final_scores, em_scores):
        """
        Computes the viterbi trellis for a given sequence.
        Receives:

        - in_scores: (num_states) array
        - trans_scores: Transition scores: (length-1, num_states, num_states) array
        - final_scores:   Final scores: (num_states) array
        - em_scores:  Emission scoress: (length, num_states) array

        :return:

        - best_path
        - best_score
        """
        # Length of the sequence.
        length = np.size(em_scores, 0)

        # Number of states
        num_states = np.size(in_scores)

        # Variables storing the Viterbi scores.
        vit_scores = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        vit_paths = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Initialization.
        vit_scores[0,:] = em_scores[0,:] + in_scores

        # Viterbi loop.
        for i in xrange(1,length):
            for state in xrange(num_states):
                vit_scores[i, state] = np.max(trans_scores[i-1, state, :] + vit_scores[i-1, :]) + em_scores[i, state]
                vit_paths[i, state] = np.argmax(trans_scores[i-1, state, :] + vit_scores[i-1, :])

        # Termination.
        best_score = np.max(vit_scores[length-1,:] + final_scores)
        best_path[length-1] = np.argmax(final_scores + vit_scores[length-1,:] )

        # Backtrack.
        for i in xrange(length-2, -1, -1):
            best_path[i] = vit_paths[i+1, best_path[i+1]]

        return best_path, best_score

    def run_forward_backward(self, initial_scores, transition_scores, final_scores, emission_scores):
        """
        Computes the forward and backguard computations

        - in_scores: (num_states) array
        - trans_scores: Transition scores: (length-1, num_states, num_states) array
        - final_scores:   Final scores: (num_states) array
        - em_scores:  Emission scoress: (length, num_states) array

        :return:
        - forward
        - backward
        """

        log_likelihood, forward = self.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        log_likelihood, backward = self.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
        print 'Log-Likelihood =', log_likelihood

        return forward, backward

