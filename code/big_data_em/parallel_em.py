from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, PickleValueProtocol
import sys
sys.path.append('..')
import numpy as np
import readers.pos_corpus as pcc
import pickle

from em_lib import *

mapping = {}
for line in open('../readers/en-ptb.map'):
    coarse,fine = line.strip().split("\t")
    mapping[coarse.lower()] = fine.lower()

word_dict, tag_dict = pickle.load(open('word_tag_dict.pkl'))
num_states = len(tag_dict)
num_observations = len(word_dict)
num_observation_labels = len(tag_dict)
num_observations = len(word_dict)
smoothing = 0.1

def load_seq(s):
    from sequences.sequence_list import SequenceList
    seq_list = SequenceList(word_dict, tag_dict)
    ex_x = []
    ex_y = []
    contents = s.decode('string-escape').split('\n')
    for line in contents:
        toks = line.split()
        if len(toks) < 2:
            continue
        pos = toks[4]
        word = toks[1]
        pos = pos.lower()

        assert pos in mapping
        assert word in word_dict

        pos = mapping[pos]
        assert pos in tag_dict

        ex_x.append(word)
        ex_y.append(pos)
    seq_list.add_sequence(ex_x, ex_y)
    return seq_list[0]

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
    def mapper(self, key, s):
        seq = load_seq(s)
        r = partial_seq(seq, self.initial_probabilities, self.transition_probabilities, self.emission_probabilities, self.final_probabilities)
        yield 'result', r

    def reducer(self, _, partials):
        res = reduce_partials(partials, smoothing)
        yield 'result', res

s = EMStep()
s.run()
