import numpy as np


def run_viterbi(initial_scores, transition_scores, final_scores, emission_scores):
    length = np.size(emission_scores, 0)  # Length of the sequence.
    num_states = np.size(initial_scores)  # Number of states.

    # Variables storing the Viterbi scores.
    viterbi_scores = np.zeros([length, num_states])

    # Variables storing the paths to backtrack.
    viterbi_paths = -np.ones([length, num_states], dtype=int)

    # Most likely sequence.
    best_path = -np.ones(length, dtype=int)

    # Initialization.
    viterbi_scores[0, :] = emission_scores[0, :] * initial_scores

    # Viterbi loop.
    for pos in xrange(1, length):
        for current_state in xrange(num_states):
            viterbi_scores[pos, current_state] = \
                np.max(viterbi_scores[pos-1, :] * transition_scores[pos-1, current_state, :])
            viterbi_scores[pos, current_state] *= emission_scores[pos, current_state]
            viterbi_paths[pos, current_state] = \
                np.argmax(viterbi_scores[pos-1, :] * transition_scores[pos-1, current_state, :])

    # Termination.
    best_score = np.max(viterbi_scores[length-1, :] * final_scores)
    best_path[length-1] = np.argmax(viterbi_scores[length-1, :] * final_scores)

    # Backtrack.
    for pos in xrange(length-2, -1, -1):
        best_path[pos] = viterbi_paths[pos+1, best_path[pos+1]]

    return best_path, best_score

# ----------
# Computes the viterbi path for a given sequence of lenght.
# N - Lenght of sequence
# H - Number of hidden states
# Receives:
# Node potentials (N,H) vector
# Edge potentials (N-1,H,H)
# ----------
# def viterbi(node_potentials,edge_potentials):
#    H,N = node_potentials.shape
#    max_marginals = np.zeros([H,N])
#
#    ## Fil backpointers with zero to signal an errors
#    backpointers = np.zeros([H,N],dtype="int")
#    backpointers[:,1:] = -1
#    max_marginals[:,0] = node_potentials[:,0]
#    for pos in xrange(1,N):
#        max_value = 0
#
#        for current_state in xrange(H):
#            max_prob = -1
#            max_state = -1
#
#            for prev_state in xrange(H):
#                viter_v = max_marginals[prev_state,pos-1]
#                trans_v = edge_potentials[prev_state,current_state,pos-1]
#                prob = viter_v*trans_v
#                if(prob > max_prob):
#                    max_prob = prob
#                    max_state = prev_state
#            max_marginals[current_state,pos] = max_prob*node_potentials[current_state,pos]
#            backpointers[current_state,pos] = max_state
#    ##Recover the viterbi state
#    viterbi_path = np.zeros([N,1],dtype="int")
#    best = np.argmax(max_marginals[:,N-1])
#    viterbi_path[N-1] = best
#    for pos in xrange(N-2,-1,-1):
# #        print viterbi_path
# #        print viterbi_path[pos+1]
# #        print "back"
# #        print backpointers
#        viterbi_path[pos] = backpointers[viterbi_path[pos+1],pos+1]
#    return viterbi_path,max_marginals
