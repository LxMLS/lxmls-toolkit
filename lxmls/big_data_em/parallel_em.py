from mrjob.job import MRJob
from mrjob.protocol import PickleProtocol, PickleValueProtocol
import numpy as np
import lxmls.readers.pos_corpus as pcc
from lxmls.sequences.hmm import HMM
import pickle
# Load the word and tag dictionaries.
word_dict, tag_dict = pickle.load(open('word_tag_dict.pkl'))

def load_sequence(s, word_dict, tag_dict):
    '''
    seq = load_sequence(s, word_dict, tag_dict)

    Load a sequence from a single line

    word_dict & tag_dict should be loaded from the file ``word_tag_dict.pkl``

    Parameters
    ----------
    s : str
    word_dict : dict
    tag_dict : dict

    Returns
    -------
    seq : Sequence object
    '''
    from lxmls.sequences.sequence_list import SequenceList
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


def predict_sequence(sequence, hmm):
    '''
    log_likelihood, initial_counts, transition_counts, final_counts,\
            emission_counts = predict_sequence(seq, hmm)

    Run forward-backward on a single sentence.

    Parameters
    ----------
    seq : Sequence object
    hmm: HMM object

    Returns
    -------
    log_likelihood : float
    initial_counts : np.ndarray
    transition_counts : ndarray
    final_counts : ndarray
    emission_counts : ndarray
    '''
    num_states = hmm.get_num_states() # Number of states.
    num_observations = hmm.get_num_observations() # Number of observation symbols.
    length = len(sequence.x) # Length of the sequence.

    # Compute scores given the observation sequence.
    initial_scores, transition_scores, final_scores, emission_scores = \
                    hmm.compute_scores(sequence)

    state_posteriors, transition_posteriors, log_likelihood = \
        hmm.compute_posteriors(initial_scores,
                               transition_scores,
                               final_scores,
                               emission_scores)

    emission_counts = {}
    initial_counts = np.zeros((num_states))
    transition_counts = np.zeros((num_states, num_states))
    final_counts = np.zeros((num_states))

    ## Take care of initial position counts.
    for y in xrange(num_states):
        initial_counts[y] += state_posteriors[0, y]

    ## Take care of emission and transition counts.
    for pos in xrange(length):
        x = sequence.x[pos]
        if x not in emission_counts:
            emission_counts[x] = np.zeros(num_states)
        for y in xrange(num_states):
            emission_counts[x][y] += state_posteriors[pos, y]
            if pos > 0:
                for y_prev in xrange(num_states):
                    transition_counts[y, y_prev] += transition_posteriors[pos-1, y, y_prev]

    ## Take care of final position counts.
    for y in xrange(num_states):
        final_counts[y] += state_posteriors[length-1, y]

    return log_likelihood, initial_counts, transition_counts, final_counts, emission_counts


def load_parameters(filename, hmm, smoothing):
    '''
    load_parameters(filename, hmm, smoothing)

    Load the HMM parameters stored in a text file.

    Parameters
    ----------
    filename : str
        Filename
    hmm : HMM object
        Will be overwritten
    smoothing : float
        Smoothing factor to use
    '''
    hmm.clear_counts(smoothing)

    f = open(filename)
    for line in f:
        if '\t' not in line:
            continue
        event, count = line.strip().split('\t')
        count = float(count)
        event = event[1:-1]
        fields = event.split(' ')
        if fields[0] == 'initial':
            y = hmm.state_labels.get_label_id(fields[1])
            hmm.initial_counts[y] += count
        elif fields[0] == 'transition':
            y = hmm.state_labels.get_label_id(fields[1])
            y_prev = hmm.state_labels.get_label_id(fields[2])
            hmm.transition_counts[y][y_prev] += count
        elif fields[0] == 'final':
            y = hmm.state_labels.get_label_id(fields[1])
            hmm.final_counts[y] += count            
        elif fields[0] == 'emission':
            x = hmm.observation_labels.get_label_id(fields[1].decode('string-escape'))
            y = hmm.state_labels.get_label_id(fields[2])
            hmm.emission_counts[x][y] += count
        else:
            continue;

    f.close()

    hmm.compute_parameters()
    

# A single iteration of the distributed EM algorithm.
class EMStep(MRJob):
    INTERNAL_PROTOCOL   = PickleProtocol
    #OUTPUT_PROTOCOL     = PickleValueProtocol
    def __init__(self, *args, **kwargs):
        MRJob.__init__(self, *args, **kwargs)
 
        # Create HMM object.
        self.hmm = HMM(word_dict, tag_dict)

        from os import path
        filename = 'parameters.txt'
        if path.exists(filename):
            # Load the HMM parameters from a text file.
            load_parameters(filename, self.hmm, smoothing=0.1)
        else:
            # Initialize the HMM parameters randomly.
            self.hmm.initialize_random()




    def mapper(self, key, s):
        seq = load_sequence(s, self.hmm.observation_labels, self.hmm.state_labels)

        log_likelihood, initial_counts, transition_counts, final_counts,\
            emission_counts = predict_sequence(seq, self.hmm)

        num_states = self.hmm.get_num_states() # Number of states.

        yield 'log-likelihood', log_likelihood
        yield 'initial', initial_counts
        for y in xrange(num_states):
            yield 'transition ' + self.hmm.state_labels.get_label_name(y), transition_counts[y,:]
        for x in emission_counts:
            yield 'emission ' + self.hmm.observation_labels.get_label_name(x), emission_counts[x]
        yield 'final', final_counts

    def reducer(self, key, counts):
        s = sum(counts)
        if key == 'log-likelihood':
            total_log_likelihood = s
            print 'Log-likelihood:', total_log_likelihood
        else:
            num_states = self.hmm.get_num_states() # Number of states.
            for y in xrange(num_states):
                yield key + ' ' + self.hmm.state_labels.get_label_name(y), s[y] 
            

            
# Run one iteration of distributed EM.
em_step = EMStep()
em_step.run()




