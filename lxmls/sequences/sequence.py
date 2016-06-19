import sys
import pdb


class Sequence(object):

    def __init__(self, sequence_list, x, y, nr):
        self.x = x
        self.y = y
        self.nr = nr
        self.sequence_list = sequence_list

    def size(self):
        """Returns the size of the sequence."""
        return len(self.x)

    def __len__(self):
        return len(self.x)

    def copy_sequence(self):
        """Performs a deep copy of the sequence"""
        s = Sequence(self.sequence_list, self.x[:], self.y[:], self.nr)
        return s

    def update_from_sequence(self, new_y):
        """Returns a new sequence equal to the previous but with y set to newy"""
        s = Sequence(self.sequence_list, self.x, new_y, self.nr)
        return s

    def __str__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%s/%s " % (self.sequence_list.x_dict.get_label_name(xi),
                               self.sequence_list.y_dict.get_label_name(yi))
        return rep

    def __repr__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            rep += "%s/%s " % (self.sequence_list.x_dict.get_label_name(xi),
                               self.sequence_list.y_dict.get_label_name(yi))
        return rep
