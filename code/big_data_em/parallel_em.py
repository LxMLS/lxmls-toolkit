from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, PickleValueProtocol
import sys
sys.path.append('..')
import numpy as np
import readers.pos_corpus as pcc

from em_lib import *

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
num_states = len(corpus.tag_dict)
num_observations = len(corpus.word_dict)
num_observation_labels = len(corpus.tag_dict)
num_observations = len(corpus.word_dict)
smoothing = 0.1

class EMStep(MRJob):
    INTERNAL_PROTOCOL   = PickleProtocol
    OUTPUT_PROTOCOL     = PickleValueProtocol
    def __init__(self, *args, **kwargs):
        MRJob.__init__(self, *args, **kwargs)
        self.initial_counts = np.zeros(num_states)

        self.emission_probabilities = np.random.random((num_observations, num_states))
        self.emission_probabilities /= self.emission_probabilities.sum(1)[:,None]

        self.initial_probabilities = np.random.random(num_states)
        self.initial_probabilities /= self.initial_probabilities.sum()

        self.final_probabilities = np.random.random(num_states)
        self.final_probabilities /= self.final_probabilities.sum()

        self.transition_probabilities = np.random.random((num_states, num_states))
        self.transition_probabilities /= self.transition_probabilities.sum(1)[:,None]
    def mapper(self, key, doci):
        seq = train_seq[int(doci)-1]
        r = partial_seq(seq, self.initial_probabilities, self.transition_probabilities, self.emission_probabilities, self.final_probabilities)
        yield 'result', r

    def reducer(self, _, partials):
        res = reduce_partials(partials, smoothing)
        yield 'result', res

s = EMStep()
s.run()
