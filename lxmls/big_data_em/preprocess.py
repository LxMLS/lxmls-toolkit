import numpy as np
import lxmls.readers.pos_corpus as pcc
import pickle

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)

pickle.dump((corpus.word_dict, corpus.tag_dict), open('word_tag_dict.pkl','w'))

with open('encoded.txt','w') as output:
    for seq in train_seq:
        words = [corpus.word_dict.get_label_name(seq.x[i]) for i in xrange(len(seq))]
        tags = [corpus.tag_dict.get_label_name(seq.y[i]) for i in xrange(len(seq))]
        s = ' '.join(['_'.join([word, tag]) for word, tag in zip(words, tags)])
        output.write(s+'\n')
        # output.write(str(seq)+'\n')

