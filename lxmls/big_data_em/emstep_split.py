from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, PickleValueProtocol
import numpy as np
import lxmls.readers.pos_corpus as pcc
from lxmls.sequences.hmm import HMM
import pickle
from emstep import load_sequence, predict_sequence, load_parameters


# A single iteration of the distributed EM algorithm.
class EMStep(MRJob):
    INTERNAL_PROTOCOL = PickleProtocol

    def __init__(self, *args, **kwargs):
        MRJob.__init__(self, *args, **kwargs)

        # Create HMM object.
        self.hmm = HMM(word_dict, tag_dict)

        from os import path
        filename = 'hmm.txt'
        if path.exists(filename):
            # Load the HMM parameters from a text file.
            load_parameters(filename, self.hmm, smoothing=0.1)
        else:
            # Initialize the HMM parameters randomly.
            self.hmm.initialize_random()

        self.log_likelihood = 0
        self.initial_counts = 0
        self.emission_counts = 0
        self.transition_counts = 0
        self.final_counts = 0

    def mapper(self, key, s):
        seq = load_sequence(s, self.hmm.observation_labels, self.hmm.state_labels)

        log_likelihood, initial_counts, transition_counts, final_counts, emission_counts = predict_sequence(
            seq, self.hmm)

        self.log_likelihood += log_likelihood
        self.initial_counts += initial_counts
        self.emission_counts += emission_counts
        self.transition_counts += transition_counts
        self.final_counts += final_counts

    def mapper_final(self):

        num_states = self.hmm.get_num_states()  # Number of states.
        num_observations = self.hmm.get_num_observations()  # Number of observation symbols.

        yield 'log-likelihood', self.log_likelihood
        for y in range(num_states):
            name_y = self.hmm.state_labels.get_label_name(y)
            for s in range(num_states):
                name_s = self.hmm.state_labels.get_label_name(s)
                yield 'transition %s %s' % (name_y, name_s), self.transition_counts[y, s]
            yield 'final '+name_y, self.final_counts[y]
            yield 'initial '+name_y, self.initial_counts[y]

        for w in range(num_observations):
            name_w = self.hmm.observation_labels.get_label_name(w)
            if self.emission_counts[w].any():
                for s in range(num_states):
                    name_s = self.hmm.state_labels.get_label_name(s)
                    if self.emission_counts[w, s]:
                        yield 'emission %s %s' % (name_w, name_s), self.emission_counts[w, s]

    def reducer(self, key, counts):
        yield key, sum(counts)


# Load the word and tag dictionaries.
word_dict, tag_dict = pickle.load(open('word_tag_dict.pkl'))

em_step = EMStep()
em_step.run()
