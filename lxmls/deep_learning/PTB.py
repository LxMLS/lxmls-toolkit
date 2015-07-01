import numpy as np
import lxmls.readers.pos_corpus as pcc
from ipdb import set_trace
from itertools import chain


# EMBEDDINGS_PATH = "/Users/samir/Downloads/glove.6B.50d.txt"
# EMB_SIZE = 50

def get_onehot(seq, trans_dict, context_size=1):
	window_idxs = []
	Y = []
	#loop over sequences and extract a context window around each word 
	for i in xrange(len(seq)):
		padded = [0]*(context_size) + translate_idxs(seq[i].x,trans_dict) + [0]* (context_size)		
		idx = [padded[j:j+(context_size*2+1)] for j in xrange(len(seq[i]))]
		# print translate_idxs(seq[i].x, trans_dict)
		# print padded
		# print idx
		# print "------------"
		# set_trace()
		window_idxs += idx
		Y += seq[i].y
	#build the bag-of-words
	X = np.zeros((len(trans_dict)+1,len(window_idxs)))
	for j in xrange(len(window_idxs)):
		# print window_idxs[j]			
		X[window_idxs[j],j] = 1			
		# set_trace()
	Y = np.array(Y)
	return X, Y

def get_embeddings(word_dict, emb_size):

	# unknown = []
	known = []
	print word_dict
	word_dict = {k.lower():v for k,v in word_dict.items()}
	print word_dict
	with open(EMBEDDINGS_PATH) as fid:		
		E = np.zeros((int(emb_size), len(word_dict)))   
		for line in fid.readlines():
			items = line.split()
			wrd   = items[0]
			# print items
			if wrd in word_dict:
				E[:, word_dict[wrd]] = np.array(items[1:]).astype(float)
				known.append(wrd)
	set_trace()

def translate_idxs(x, translation_dict):

	return [translation_dict[idx] for idx in x]

def read_corpora():
	corpus    = pcc.PostagCorpus()
	train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll",max_sent_len=10,max_nr_sent=1000)
	test_seq  = corpus.read_sequence_list_conll("data/test-23.conll",max_sent_len=10,max_nr_sent=1000)
	dev_seq   = corpus.read_sequence_list_conll("data/dev-22.conll",max_sent_len=10,max_nr_sent=1000)

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
	#data in one hot matrices
	train_seq_X, train_seq_Y = get_onehot(train_seq, translation_dict)
	test_seq_X, test_seq_Y = get_onehot(test_seq, translation_dict)

	return train_seq_X, train_seq_Y, test_seq_X, test_seq_Y

