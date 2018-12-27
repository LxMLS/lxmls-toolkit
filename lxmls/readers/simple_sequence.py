import sys
from lxmls.sequences.label_dictionary import *
from lxmls.sequences.sequence import *
from lxmls.sequences.sequence_list import *


# ----------
# Implements a simple sequence example taken from wikipedia
# States are Sunny and Rainy
# Observations are: Shop, Clean, Walk
# ----------

class SimpleSequence:

    def __init__(self):
        # Observation set.
        self.x_dict = LabelDictionary(['walk', 'shop', 'clean', 'tennis'])

        # State set.
        self.y_dict = LabelDictionary(['rainy', 'sunny'])

        # Generate training sequences.
        train_sequences = SequenceList(self.x_dict, self.y_dict)
        train_sequences.add_sequence(['walk', 'walk', 'shop', 'clean'], ['rainy', 'sunny', 'sunny', 'sunny'])
        train_sequences.add_sequence(['walk', 'walk', 'shop', 'clean'], ['rainy', 'rainy', 'rainy', 'sunny'])
        train_sequences.add_sequence(['walk', 'shop', 'shop', 'clean'], ['sunny', 'sunny', 'sunny', 'sunny'])

        # Generate test sequences.
        test_sequences = SequenceList(self.x_dict, self.y_dict)
        test_sequences.add_sequence(['walk', 'walk', 'shop', 'clean'], ['rainy', 'sunny', 'sunny', 'sunny'])
        test_sequences.add_sequence(['clean', 'walk', 'tennis', 'walk'], ['sunny', 'sunny', 'sunny', 'sunny'])

        self.train = train_sequences
        self.test = test_sequences
