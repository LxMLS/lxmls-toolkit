import numpy as np


# ----------
# Computes the viterbi path for a given sequence of lenght.
# N - Lenght of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
# ----------
def viterbi(node_potentials, edge_potentials):
    H, N = node_potentials.shape
    max_marginals = np.zeros([H, N])

    # Fill backpointers with zero to signal an errors
    backpointers = np.zeros([H, N], dtype="int")
    backpointers[:, 1:] = -1
    max_marginals[:, 0] = log_stable(node_potentials[:, 0])
    for pos in xrange(1, N):
        max_value = 0

        for current_state in xrange(H):
            max_logprob = -1000
            max_state = -1

            for prev_state in xrange(H):
                viter_v = max_marginals[prev_state, pos-1]
                trans_v = log_stable(edge_potentials[prev_state, current_state, pos-1])
                logprob = viter_v + trans_v
                if logprob > max_logprob:
                    max_logprob = logprob
                    max_state = prev_state
            max_marginals[current_state, pos] = max_logprob + log_stable(node_potentials[current_state, pos])
            backpointers[current_state, pos] = max_state
    # Recover the viterbi state
    viterbi_path = np.zeros([N, 1], dtype="int")
    best = np.argmax(max_marginals[:, N-1])
    viterbi_path[N-1] = best
    for pos in xrange(N-2, -1, -1):
        #        print viterbi_path
        #        print viterbi_path[pos+1]
        #        print "back"
        #        print backpointers
        viterbi_path[pos] = backpointers[viterbi_path[pos+1], pos+1]
    return viterbi_path, max_marginals


# ----------
# Computes the viterbi path for a given sequence of lenght in log space. (node and edge features come in log space)
# N - Lenght of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
# ----------
def viterbi_log(node_potentials, edge_potentials):
    H, N = node_potentials.shape
    max_marginals = np.zeros([H, N])

    # Fill backpointers with zero to signal an errors
    backpointers = np.zeros([H, N], dtype="int")
    backpointers[:, 1:] = -1
    max_marginals[:, 0] = node_potentials[:, 0]
    for pos in xrange(1, N):
        max_value = 0

        for current_state in xrange(H):
            max_logprob = -1000
            max_state = -1

            for prev_state in xrange(H):
                viter_v = max_marginals[prev_state, pos-1]
                trans_v = edge_potentials[prev_state, current_state, pos-1]
                logprob = viter_v + trans_v
                if logprob > max_logprob:
                    max_logprob = logprob
                    max_state = prev_state
            max_marginals[current_state, pos] = max_logprob + node_potentials[current_state, pos]
            backpointers[current_state, pos] = max_state
    # Recover the viterbi state
    viterbi_path = np.zeros([N, 1], dtype="int")
    best = np.argmax(max_marginals[:, N-1])
    viterbi_path[N-1] = best
    for pos in xrange(N-2, -1, -1):
        #        print viterbi_path
        #        print viterbi_path[pos+1]
        #        print "back"
        #        print backpointers
        viterbi_path[pos] = backpointers[viterbi_path[pos+1], pos+1]
    return viterbi_path, max_marginals


def log_stable(x):
    #    if x < 1e-50:
    #        return -50
    #    elif x > 1e50:
    #        return 50
    #    else:
    return np.log(x)
