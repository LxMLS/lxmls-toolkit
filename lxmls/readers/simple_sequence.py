import sys
from sequences.sequence import *
from sequences.sequence_list import *


################
### Implements a simple sequence example taken from wikipedia
### States are Sunny and Rainy
### Observations are: Shop, Clean, Walk
#################

class SimpleSequence:

    def __init__(self):
        y_dict = {'r' : 0, 's': 1}
        int_to_pos = ['r','s']
        x_dict = {'w' : 0, 's' : 1, 'c' : 2, 't' : 3}
        int_to_word = ['w','s','c','t']
        sl = Sequence_List(x_dict,int_to_word,y_dict,int_to_pos)
        sl2 = Sequence_List(x_dict,int_to_word,y_dict,int_to_pos)
        sl.add_sequence([0,0,1,2],[0,1,1,1])
        sl.add_sequence([0,0,1,2],[0,0,0,1])
        sl.add_sequence([0,1,1,2],[1,1,1,1])
        sl2.add_sequence([0,0,1,2],[0,1,1,1])
        sl2.add_sequence([2,0,3,0],[1,1,1,1])

        self.x_dict = x_dict
        self.y_dict = y_dict
        self.int_to_word = int_to_word
        self.int_to_pos = int_to_pos
        self.train = sl
        self.dev = 0
        self.test = sl2
