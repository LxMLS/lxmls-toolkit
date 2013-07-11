from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, PickleValueProtocol
import sys
sys.path.append('.')
import numpy as np
import readers.pos_corpus as pcc
import pickle

from em_lib import *
from sequences.hmm import HMM


# Function to load a sequence from a single line.
def load_seq(s):
    from sequences.sequence_list import SequenceList
    seq_list = SequenceList(word_dict, tag_dict)
    words = []
    tags = []
    line = s.rstrip()
    pairs = line.split(' ')

    for pair in pairs:
        fields = pair.split('_')
        words.append(fields[0])
        tags.append(fields[1])

    seq_list.add_sequence(words, tags)
    return seq_list[0]


# A single iteration of the distributed EM algorithm.
class EMStep(MRJob):
    INTERNAL_PROTOCOL   = PickleProtocol
    OUTPUT_PROTOCOL     = PickleValueProtocol
    def __init__(self, *args, **kwargs):
        MRJob.__init__(self, *args, **kwargs)
        self.hmm = hmm

    def mapper(self, key, s):
        seq = load_seq(s)
        r = partial_seq(seq, self.hmm)
        yield 'result', r

    def reducer(self, _, partials):
        total_log_likelihood = reduce_partials(partials, self.hmm, smoothing)
        print 'Log-likelihood:', total_log_likelihood
        yield 'result', _ 



# Load the word and tag dictionaries.
word_dict, tag_dict = pickle.load(open('word_tag_dict.pkl'))

# Initialize the HMM parameters randomly.
hmm = HMM(word_dict, tag_dict)
hmm.initialize_random()

# Set the smoothing coefficient.
smoothing = 0.1

# Run 10 iterations of distributed EM.
em_step = EMStep()
for iteration in xrange(10):
    em_step.run()

# Evaluate the final model.
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
acc = hmm.evaluate_EM(train_seq)
print acc



