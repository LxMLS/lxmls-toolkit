import numpy as np


# ----------
# Computes the forward backward trellis for a given sequence.
# N - Length of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
#
# Emission probabilities: (length, num_states) array
# Initial probabilities: (num_states) array
# Transition probabilities: (length, num_states+1, num_states) array
#
# OR
#
# Transition probabilities: (length-1, num_states, num_states) array
# Final probabilities: (num_states) array
# ----------
def run_forward(initial_scores, transition_scores, final_scores, emission_scores):
    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Forward variables.
    forward = np.zeros([length, num_states])

    # Initialization.
    forward[0, :] = emission_scores[0, :] * initial_scores

    # Forward loop.
    for pos in xrange(1, length):
        for current_state in xrange(num_states):
            forward[pos, current_state] = \
                np.sum(forward[pos-1, :] * transition_scores[pos-1, current_state, :])
            forward[pos, current_state] *= emission_scores[pos, current_state]

    # Termination.
    likelihood = sum(forward[length-1, :] * final_scores)
    #    print 'Likelihood =', likelihood
    return likelihood, forward


def run_backward(initial_scores, transition_scores, final_scores, emission_scores):
    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Backward variables.
    backward = np.zeros([length, num_states])

    # Initialization.
    backward[length-1, :] = final_scores

    # Backward loop.
    for pos in xrange(length-2, -1, -1):
        for current_state in xrange(num_states):
            backward[pos, current_state] = \
                sum(backward[pos+1, :] *
                    transition_scores[pos, :, current_state] *
                    emission_scores[pos+1, :])
            #            prob = 0.0
            #            for next_state in xrange(num_states):
            #                back = backward[pos+1, next_state]
            #                trans = transition_scores[pos, next_state, current_state];
            #                observation = emission_scores[pos+1, next_state];
            #                prob += trans * observation * back;
            #            backward[pos, current_state] = prob
            #    backward[0,:] *= initial_scores
    # sanity_check_forward_backward(forward,backward)

    # Termination.
    likelihood = sum(backward[0, :] * initial_scores*emission_scores[0, :])
    #    print 'Likelihood =', likelihood
    return likelihood, backward


def forward_backward(initial_scores, transition_scores, final_scores, emission_scores):
    likelihood, forward = run_forward(initial_scores, transition_scores, final_scores, emission_scores)
    print 'Likelihood =', likelihood

    likelihood, backward = run_backward(initial_scores, transition_scores, final_scores, emission_scores)
    print 'Likelihood =', likelihood

    #    length = np.size(emission_scores, 0) # Length of the sequence.
    #    num_states = np.size(initial_scores) # Number of states.
    #
    #    forward = np.zeros([length, num_states])
    #    backward = np.zeros([length, num_states])
    #
    #    forward[0,:] = emission_scores[0,:] * initial_scores
    #    # Forward loop.
    #    for pos in xrange(1,length):
    #        for current_state in xrange(num_states):
    #            for prev_state in xrange(num_states):
    #                forward_v = forward[pos-1, prev_state]
    #                trans_v = transition_scores[pos-1, current_state, prev_state]
    #                prob = forward_v*trans_v
    #                forward[pos, current_state] += prob
    #            forward[pos, current_state] *= emission_scores[pos, current_state]
    #    # forward[length-1,:] *= final_scores
    #    print 'Likelihood =', sum(forward[length-1,:] * final_scores)
    #
    #    # Backward loop.
    #    # backward[length-1,:] = final_scores
    #    backward[length-1,:] = final_scores #1.0
    #    for pos in xrange(length-2,-1,-1):
    #        for current_state in xrange(num_states):
    #            prob = 0.0
    #            for next_state in xrange(num_states):
    #                back = backward[pos+1, next_state]
    #                trans = transition_scores[pos, next_state, current_state];
    #                observation = emission_scores[pos+1, next_state];
    #                prob += trans * observation * back;
    #            backward[pos, current_state] = prob
    #    # backward[0,:] *= initial_scores
    #    # sanity_check_forward_backward(forward,backward)
    #    print 'Likelihood =', sum(backward[0,:] * initial_scores * emission_scores[0,:])

    return forward, backward


# ----------
# Computes the forward backward trellis for a given sequence and node and edge potentials
# N - Length of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
# ----------
# def forward_backward(node_potentials,edge_potentials):
#    H,N = node_potentials.shape
#    forward = np.zeros([H,N],dtype=float)
#    backward = np.zeros([H,N],dtype=float)
#    forward[:,0] = node_potentials[:,0]
#    # Forward loop
#    for pos in xrange(1,N):
#        for current_state in xrange(H):
#            for prev_state in xrange(H):
#                forward_v = forward[prev_state,pos-1]
#                trans_v = edge_potentials[prev_state,current_state,pos-1]
#                prob = forward_v*trans_v
#                forward[current_state,pos] += prob
#            forward[current_state,pos] *= node_potentials[current_state,pos]
#    # Backward loop
#    backward[:,N-1] = 1
#    for pos in xrange(N-2,-1,-1):
#        for current_state in xrange(H):
#            prob = 0
#            for next_state in xrange(H):
#                back = backward[next_state,pos+1]
#                trans = edge_potentials[current_state,next_state,pos];
#                observation = node_potentials[next_state,pos+1];
#                prob += trans * observation * back;
#            backward[current_state,pos] = prob
#    # sanity_check_forward_backward(forward,backward)
#    return forward,backward


# def forward_backward_trans_probs(node_potentials,transitions_probs):
#     H,N = node_potentials.shape
#     forward = np.zeros([H,N],dtype=float)
#     backward = np.zeros([H,N],dtype=float)
#     forward[:,0] = node_potentials[:,0]
#     # Forward loop
#     for pos in xrange(1,N):
#         for current_state in xrange(H):
#             for prev_state in xrange(H):
#                 forward_v = forward[prev_state,pos-1]
#                 trans_v = transitions_probs[current_state,prev_state]
#                 prob = forward_v*trans_v
#                 forward[current_state,pos] += prob
#             forward[current_state,pos] *= node_potentials[current_state,pos]
#     # Backward loop
#     backward[:,N-1] = 1
#     for pos in xrange(N-2,-1,-1):
#         for current_state in xrange(H):
#             prob = 0
#             for next_state in xrange(H):
#                 back = backward[next_state,pos+1]
#                 trans = transition_probs[next_state,current_state];
#                 observation = node_potentials[next_state,pos+1];
#                 prob += trans * observation * back;
#             backward[current_state,pos] = prob
#     # sanity_check_forward_backward(forward,backward)
#     return forward,backward


# ----------
# For every position - pos the sum_states forward(pos,state)*backward(pos,state) = Likelihood
# ----------
def sanity_check_forward_backward(forward, backward):
    N, H = forward.shape
    likelihood = np.zeros([N, 1])
    for pos in xrange(N):
        aux = 0
        for current_state in xrange(H):
            aux += forward[pos, current_state] * backward[pos, current_state]
        likelihood[pos] = aux
        for i in xrange(pos):
            if abs(aux - likelihood[i]) > 0.001:
                print "Likelihood for pos %i and pos %i mismatch: %f - %f" % (i, pos, likelihood[i], aux)
                return False
    print likelihood
    return True
