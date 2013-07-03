import sequences.sequence_classification_decoder as scd
import numpy as np
def partial_seq(sequence, initial_probabilities, transition_probabilities, emission_probabilities, final_probabilities):
    length = len(sequence.x)
    num_observations, num_states = emission_probabilities.shape

    emission_scores = np.zeros([length, num_states]) 
    initial_scores = np.zeros(num_states) 
    transition_scores = np.zeros([length-1, num_states, num_states])
    final_scores = np.zeros(num_states)

    initial_scores[:] = np.log(initial_probabilities)

    emission_scores = np.zeros([length, num_states]) 

    for pos in xrange(length):
        emission_scores[pos,:] = np.log(emission_probabilities[sequence.x[pos], :])
        if pos > 0:
            transition_scores[pos-1,:,:] = np.log(transition_probabilities)
            
    decoder = scd.SequenceClassificationDecoder()
    _, forward = decoder.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
    log_likelihood, backward = decoder.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
    state_posteriors = (forward + backward - log_likelihood)
        
    transition_posteriors = np.zeros([length-1, num_states, num_states])
    for pos in xrange(length-1):
        for prev_state in xrange(num_states):
            for state in xrange(num_states):
                transition_posteriors[pos, state, prev_state] = \
                                        forward[pos, prev_state] + \
                                        transition_scores[pos, state, prev_state] + \
                                        emission_scores[pos+1, state] + \
                                        backward[pos+1, state]
                transition_posteriors[pos, state, prev_state] -= log_likelihood
                
    state_posteriors = np.exp(state_posteriors)
    transition_posteriors = np.exp(transition_posteriors)
    initial_counts = state_posteriors[0]
    emission_counts = np.zeros((num_observations, num_states))
    transition_counts = np.zeros((num_states, num_states))
    for pos in xrange(length):
        x = sequence.x[pos]
        for y in xrange(num_states):
            emission_counts[x, y] += state_posteriors[pos, y]
            if pos > 0:
                for y_prev in xrange(num_states):
                    transition_counts[y, y_prev] += transition_posteriors[pos-1, y, y_prev]
    final_counts = np.zeros(final_probabilities.shape)
    for y in xrange(num_states):
        final_counts[y] += state_posteriors[length-1, y]
    return log_likelihood, initial_counts, transition_counts, emission_counts, final_counts

def reduce_partials(partials, smoothing):
    partials = list(partials)
    res = []
    for pi in xrange(5):
        res.append(sum(p[pi] for p in partials))
    total_log_likelihood,initial_counts, transition_counts, emission_counts,final_counts = res
    initial_counts += smoothing
    transition_counts += smoothing
    final_counts += smoothing
    emission_counts += smoothing

    initial_counts /= initial_counts.sum()
    sum_transition = np.sum(transition_counts, 0) + final_counts
    num_observations, num_states = emission_counts.shape

    transition_probabilities = transition_counts / np.tile(sum_transition, [num_states, 1])
    final_probs = final_counts / sum_transition

    sum_emission = np.sum(emission_counts, 0)
    emission_probabilities = emission_counts / np.tile(sum_emission, [num_observations, 1])
    return total_log_likelihood,initial_counts, transition_probabilities, emission_probabilities,final_probs


def compute_scores(sequence, emission_probabilities, transition_probabilities):
    length = len(sequence)
    num_observations, num_states = emission_probabilities.shape

    emission_scores = np.zeros([length, num_states]) 
    initial_scores = np.zeros(num_states) 
    transition_scores = np.zeros([length-1, num_states, num_states])
    final_scores = np.zeros(num_states)

    initial_scores[:] = np.log(initial_probabilities)

    emission_scores = np.zeros([length, num_states]) 

    for pos in xrange(length):
        emission_scores[pos,:] = np.log(emission_probabilities[sequence.x[pos], :])
        if pos > 0:
            transition_scores[pos-1,:,:] = np.log(transition_probabilities)
    return initial_scores, transition_scores, final_scores, emission_scores
