import numpy as np


# ----------
# Computes the forward backward trellis for a given sequence
# N - Lenght of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
# ----------
def forward_backward(node_potentials, edge_potentials):
    H, N = node_potentials.shape
    forward = -1000.0 * np.ones([H, N], dtype=float)
    backward = -1000.0 * np.ones([H, N], dtype=float)
    forward[:, 0] = np.log(node_potentials[:, 0])
    # Forward loop
    for pos in xrange(1, N):
        for current_state in xrange(H):
            for prev_state in xrange(H):
                forward_v = forward[prev_state, pos-1]
                trans_v = np.log(edge_potentials[prev_state, current_state, pos-1])
                logprob = forward_v + trans_v
                forward[current_state, pos] = np.logaddexp(forward[current_state, pos], logprob)
            forward[current_state, pos] += np.log(node_potentials[current_state, pos])
    # Backward loop
    backward[:, N-1] = 0.0  # log(1) = 0
    for pos in xrange(N-2, -1, -1):
        for current_state in xrange(H):
            logprob = -1000.0
            for next_state in xrange(H):
                back = backward[next_state, pos+1]
                trans = np.log(edge_potentials[current_state, next_state, pos])
                observation = np.log(node_potentials[next_state, pos+1])
                logprob = np.logaddexp(logprob, trans + observation + back)
            backward[current_state, pos] = logprob
    # sanity_check_forward_backward(forward,backward)
    # print forward, backward
    return np.exp(forward), np.exp(backward)


# ----------
# For every position - pos the sum_states forward(pos,state)*backward(pos,state) = Likelihood
# ----------
def sanity_check_forward_backward(forward, backward):
    H, N = forward.shape
    likelihood = np.zeros([N, 1])
    for pos in xrange(N):
        aux = 0
        for current_state in xrange(H):
            aux += forward[current_state, pos] * backward[current_state, pos]
        likelihood[pos] = aux
    return likelihood
