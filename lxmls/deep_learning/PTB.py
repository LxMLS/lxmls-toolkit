import numpy as np
import lxmls.readers.pos_corpus as pcc
from ipdb import set_trace
from itertools import chain

def get_onehot(seq, x_dim, trans_dict):
	
	padded_x = [translate_idxs(seq[i].x,trans_dict) + [0]*(x_dim - len(seq[i])) for i in xrange(len(seq))]
	padded_y = [seq[i].y + [0]*(x_dim - len(seq[i])) for i in xrange(len(seq))]
	
	return np.matrix(padded_x).T, np.matrix(padded_y).T
	
def translate_idxs(x, translation_dict):

	return [translation_dict[idx] for idx in x]
	
# TODO: We need to compacify dictionaries to remove words not present in the
# pruned data, include context, have single monolitic x, y matrices
corpus    = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
test_seq  = corpus.read_sequence_list_conll("data/test-23.conll",max_sent_len=15,max_nr_sent=1000)
dev_seq   = corpus.read_sequence_list_conll("data/dev-22.conll",max_sent_len=15,max_nr_sent=1000)

#collect word indices
idx  = []
idx += [train_seq[i].x for i in xrange(len(train_seq))]
idx += [test_seq[i].x for i in xrange(len(test_seq))]
idx += [dev_seq[i].x for i in xrange(len(dev_seq))]
unique_idx = list(set(chain(*idx)))
old_dict = train_seq.x_dict
#inverse dictionary
old_dict_inv = {v:k for k,v in old_dict.items()}
#reduced dictionary
reduced_dict = {old_dict_inv[unique_idx[i]] : i+1 for i in xrange(len(unique_idx))}	
reduced_dict['_PADD_'] = 0
#translation dictionary
translation_dict = {old_dict[word]:reduced_dict[word] for word in reduced_dict.keys() if word in old_dict}
#data in one hot matrix
train_seq_X, train_seq_Y = get_onehot(train_seq, 15, translation_dict)
set_trace()