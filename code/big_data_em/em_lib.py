from scipy import sparse
import sequences.hmm as hmmc
import numpy as np

def partial_seq(sequence, hmm):

    num_states = hmm.get_num_states() # Number of states.
    num_observations = hmm.get_num_observations() # Number of observation symbols.
    length = len(sequence.x) # Length of the sequence.

    # Compute scores given the observation sequence.
    initial_scores, transition_scores, final_scores, emission_scores = \
                    hmm.compute_scores(sequence)

    state_posteriors, transition_posteriors, log_likelihood = \
        hmm.compute_posteriors(initial_scores,
                               transition_scores,
                               final_scores,
                               emission_scores)

    emission_counts = {} #np.zeros((num_observations, num_states))
    initial_counts = np.zeros((num_states))
    transition_counts = np.zeros((num_states, num_states))
    final_counts = np.zeros((num_states))

    ## Take care of initial position counts.
    for y in xrange(num_states):
        initial_counts[y] += state_posteriors[0, y]

    ## Take care of emission and transition counts.
    for pos in xrange(length):
        x = sequence.x[pos]
        if x not in emission_counts:
            emission_counts[x] = np.zeros(num_states)
        for y in xrange(num_states):
            emission_counts[x][y] += state_posteriors[pos, y]
            if pos > 0:
                for y_prev in xrange(num_states):
                    transition_counts[y, y_prev] += transition_posteriors[pos-1, y, y_prev]

    ## Take care of final position counts.
    for y in xrange(num_states):
        final_counts[y] += state_posteriors[length-1, y]

    return log_likelihood, initial_counts, transition_counts, final_counts, emission_counts


def reduce_partials(partials, hmm, smoothing):

    num_states = hmm.get_num_states() # Number of states.
    num_observations = hmm.get_num_observations() # Number of observation symbols.

    hmm.clear_counts(smoothing)

    partials = list(partials)

    total_log_likelihood = 0.0

    for p in partials:
        total_log_likelihood += p[0]
        hmm.initial_counts += p[1]
        hmm.transition_counts += p[2]
        hmm.final_counts += p[3]
        hmm.emission_counts += p[4].todense()

    hmm.compute_parameters()

    return total_log_likelihood



