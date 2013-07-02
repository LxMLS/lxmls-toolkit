import sys
sys.path.append('..')
import numpy as np
import readers.pos_corpus as pcc

from em_lib import *

import os
os.chdir('..')

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
num_states = len(corpus.tag_dict)
num_observations = len(corpus.word_dict)
num_observation_labels = len(corpus.tag_dict)
num_observations = len(corpus.word_dict)
initial_counts = np.zeros(num_states)

emission_probabilities = np.random.random((num_observations, num_states))
emission_probabilities /= emission_probabilities.sum(1)[:,None]

initial_probabilities = np.random.random(num_states)
initial_probabilities /= initial_probabilities.sum()

final_probabilities = np.random.random(num_states)
final_probabilities /= final_probabilities.sum()

transition_probabilities = np.random.random((num_states, num_states))
transition_probabilities /= transition_probabilities.sum(1)[:,None]
smoothing = 0.1

for epoch in xrange(20):
    partials = []
    for seq in train_seq:
        partials.append(partial_seq(seq, initial_probabilities, transition_probabilities, emission_probabilities, final_probabilities))

    total_log_likelihood,initial_probabilities, transition_probabilities, emission_probabilities,final_probabilities = reduce_partials(partials, smoothing)
    print initial_probabilities
    print epoch+1, total_log_likelihood
