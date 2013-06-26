import sequence as seq
import pdb

class SequenceList:
    def __init__(self, x_dict, y_dict):
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.seq_list = []

    def __str__(self):
        return str(self.seq_list)

    def __repr__(self):
        return repr(self.seq_list)


    def size(self):
        '''Returns the number of sequences in the list.'''
        return len(self.seq_list)
        
    def get_num_tokens(self):
        '''Returns the number of tokens in the sequence list, that is, the 
        sum of the length of the sequences.'''
        return sum([seq.size() for seq in self.seq_list])
        
    def add_sequence(self, x, y):
        '''Add a sequence to the list, where x is the sequence of
        observations, and y is the sequence of states.'''
        num_seqs = len(self.seq_list) 
        x_ids = [self.x_dict.get_label_id(name) for name in x]
        y_ids = [self.y_dict.get_label_id(name) for name in y]
        self.seq_list.append(seq.Sequence(self, x_ids, y_ids, num_seqs))
        

    def save(self,file):
        seq_fn = open(file,"w")
        for seq in self.seq_list:
            txt = ""
            for pos,word in enumerate(seq.x):
                txt += "%i:%i\t"%(word,seq.y[pos])
            seq_fn.write(txt.strip()+"\n")
        seq_fn.close()

    def load(self,file):
        seq_fn = open(file,"r")
        seq_list = []
        for line in seq_fn:
            seq_x = []
            seq_y = []
            entries = line.strip().split("\t")
            for entry in entries:
                x,y = entry.split(":")
                seq_x.append(int(x))
                seq_y.append(int(y))
            self.add_sequence(seq_x,seq_y)
        seq_fn.close()
        
#    ## Returns all sentences with a given word
#    def find_sentences_with_word(self,word):
#        seq_idx = []
#        target_w = self.x_dict[word]
#        for sequence in self.seq_list:
#            for word in sequence.x:
#                if word == target_w:
#                    seq_idx.append(sequence.nr)
#        return seq_idx
