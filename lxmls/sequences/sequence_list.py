from . import sequence as seq


class Sequence_List:
    def __init__(self,x_dict,int_to_word,y_dict,int_to_pos):
        self.x_dict = x_dict;
        self.int_to_word = int_to_word
        self.seq_list = []
        self.y_dict = y_dict
        self.int_to_pos = int_to_pos
        self.nr_seqs = 0


    def add_sequence(self,x,y):
        self.seq_list.append(seq.Sequence(self,x,y,self.nr_seqs))
        self.nr_seqs +=1
        
    def __str__(self):
        return str(self.seq_list)

    def __repr__(self):
        return repr(self.seq_list)


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
        
    ## Returns all sentences with a given word
    def find_sentences_with_word(self,word):
        seq_idx = []
        target_w = self.x_dict[word]
        for sequence in self.seq_list:
            for word in sequence.x:
                if word == target_w:
                    seq_idx.append(sequence.nr)
        return seq_idx
