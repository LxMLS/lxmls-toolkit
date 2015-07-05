#!/usr/bin/python
import cPickle  # To store classes on files
# DEBUG
from ipdb import set_trace
import numpy as np
import os

#DATASETS
tweets_train_path = 'DATA/twitter/datasets/semeval_train.txt'  
tweets_2013_path  = 'DATA/twitter/datasets/tweets_2013.txt'
tweets_2014_path  = 'DATA/twitter/datasets/tweets_2014.txt'
tweets_2015_path  = 'DATA/twitter/datasets/tweets_2015.txt'

#TODO: find a place to put the embeddings 
#TODO: create a pruned version of the embeddings file containing only the needed vectors
emb_path = '/Users/samir/Code/resources/WordModels/Embeddings/str_skip_600.txt'    

# VOCABULARY FEATURES 
train_corpus_feat = 'DATA/twitter/features/semeval-pretokenized.pkl' 
eval_2013_feat    = 'DATA/twitter/features/official_tweets_2013.pkl' 
eval_2014_feat    = 'DATA/twitter/features/official_tweets_2014.pkl'
eval_2015_feat    = 'DATA/twitter/features/official_tweets_2015.pkl'
pretrained_emb    = 'DATA/twitter/features/E.pkl'

def FmesSemEval(gold, pred):
    '''
    Compute SemEval metric
    Average F-measure of the positive and negative classes
    '''

    # Confusion Matrix
    # This assumes the order (neut-sent, pos-sent, neg-sent)
    mapp     = np.array([ 1, 2, 0])
    conf_mat = np.zeros((3, 3))
    for y, hat_y in zip(gold, pred):        
        conf_mat[mapp[y], mapp[hat_y]] += 1
    
    # POS-SENT 
    # True positives pos-sent
    tp = conf_mat[1, 1]
    # False postives pos-sent
    fp = conf_mat[:, 1].sum() - tp
    # False engatives pos-sent
    fn = conf_mat[1, :].sum() - tp
    # Fmeasure binary
    FmesPosSent = Fmeasure(tp, fp, fn)

    # NEG-SENT 
    # True positives pos-sent
    tp = conf_mat[2, 2]
    # False postives pos-sent
    fp = conf_mat[:, 2].sum() - tp
    # False engatives pos-sent
    fn = conf_mat[2, :].sum() - tp
    # Fmeasure binary
    FmesNegSent = Fmeasure(tp, fp, fn)
 
    return (FmesPosSent + FmesNegSent)/2

def Fmeasure(tp, fp, fn):
    # Precision
    if tp+fp:
        precision = tp/(tp+fp)
    else:
        precision = 0 
    # Recall
    if tp+fn:
        recall    = tp/(tp+fn)
    else:
        recall    = 0
    # F-measure
    if precision + recall:
        return 2 * (precision * recall)/(precision + recall)
    else:
        return 0 

def split_train_dev(train_x, train_y, perc=0.8):
    '''
    Split train set into train and dev
    '''

    # RANDOM SEED
    rng = np.random.RandomState(1234)

    # Ensure data is suitable for theano
    data_x = train_x #[tx.astype('int32') for tx in train_x]
    data_y = train_y #[np.array(ty).astype('int32')[None] for ty in train_y]
    # Separate into the different classes
    data_pos_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 0] 
    data_neg_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 1] 
    data_neu_x = [data_x[i] for i in range(len(data_y)) if data_y[i] == 2] 
    data_pos_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 0] 
    data_neg_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 1] 
    data_neu_y = [data_y[i] for i in range(len(data_y)) if data_y[i] == 2] 
    # Divide into train/dev mantaining observed class distribution
    L_train_pos = int(len(data_pos_x)*perc)
    L_train_neg = int(len(data_neg_x)*perc)
    L_train_neu = int(len(data_neu_x)*perc)
    L_train     = L_train_pos + L_train_neg + L_train_neu
    # Compose datasets
    train_x = (data_pos_x[:L_train_pos] + data_neg_x[:L_train_neg] 
               + data_neu_x[:L_train_neu])
    train_y = (data_pos_y[:L_train_pos] + data_neg_y[:L_train_neg] 
               + data_neu_y[:L_train_neu])
    dev_x   = (data_pos_x[L_train_pos:] + data_neg_x[L_train_neg:] 
               + data_neu_x[L_train_neu:])
    dev_y   = (data_pos_y[L_train_pos:] + data_neg_y[L_train_neg:] 
               + data_neu_y[L_train_neu:])
    # Shuffle them
    train_idx = np.arange(len(train_x))
    rng.shuffle(train_idx)
    train_x   = [train_x[i] for i in train_idx]
    train_y   = np.array([train_y[i] for i in train_idx])
    dev_idx   = np.arange(len(dev_x))
    rng.shuffle(dev_idx)
    dev_x     = [dev_x[i] for i in dev_idx]
    dev_y     = np.array([dev_y[i] for i in dev_idx])
    
    return train_x, train_y, dev_x, dev_y 

def extract_feats(corpus, wrd2idx):
    '''
    Convert semeval corpus into binary format    
    '''
    # Extract data into matrix, take into account max size
    x = [] 
    y = []

    n_in  = 0
    n_out = 0

    for tweet in corpus:
        # ONE-HOT WORD FEATURES
        tmp_x = []
        for wrd in tweet[1]:
             if wrd in wrd2idx:
                 tmp_x.append(wrd2idx[wrd])
                 n_in += 1
             else:
                 # UNKNOWN
                 tmp_x.append(1)
                 n_out += 1        
        x.append(tmp_x)
        # TARGETS
        if tweet[0] == 'positive':
            y.append(0)
        elif tweet[0] == 'negative':
            y.append(1)
        elif tweet[0] == 'neutral':
            y.append(2)
        else:
            raise ValueError, ("Unexpected Label! %s" % tweet[0])
            
    return np.array(x), np.array(y)

def read_corpus(corpus_path):

    with open(corpus_path) as f:
        corpus = [(line.split()[2], line.split()[4:]) for line in f.readlines()]

    return corpus

class SemEvalReader:

    def __init__(self):
        
        train_raw = read_corpus(tweets_train_path)
        eval2013  = read_corpus(tweets_2013_path)
        eval2014  = read_corpus(tweets_2014_path)
        eval2015  = read_corpus(tweets_2015_path)
        #GET DICTIONARY FOR ALL CORPORA
        self.wrd2idx = {}
        idx      = 0
        for tweet in train_raw + eval2013 + eval2014 + eval2015:
            for wrd in tweet[1]:
                if wrd not in self.wrd2idx:
                    self.wrd2idx[wrd]  = idx
                    idx          += 1
        self.voc_size = len(self.wrd2idx)       
        #EXTRACT FEATURES
        train_raw_x, train_raw_y = extract_feats(train_raw, self.wrd2idx) 
        #shuffle traininig data and split into train and dev
        train_x, train_y, dev_x, dev_y = split_train_dev(train_raw_x, train_raw_y, perc=0.8)
        self.train      = train_x, train_y
        self.dev        = dev_x, dev_y  
        self.tweets2013 = extract_feats(eval2013, self.wrd2idx)
        self.tweets2014 = extract_feats(eval2014, self.wrd2idx)
        self.tweets2015 = extract_feats(eval2015, self.wrd2idx)
 
    def get_bow(self, dataset):

        '''
            Return the dataset as a bag-of-words matrix
        '''
        if dataset == "train":
            X, _ = self.train
        elif dataset == "dev":
            X, _ = self.dev
        elif dataset == "tweets2013":
            X, _ = self.tweets2013
        elif dataset == "tweets2014":
            X, _ = self.tweets2014
        elif dataset == "tweets2015":
            X, _ = self.tweets2015
        else:
            raise ValueError, ("Unknown dataset: %s" % dataset)

        bow = np.zeros((self.voc_size, len(X)))
        for i, x in enumerate(X):
            bow[x,i] = 1
            
        return bow

    def get_embedding(self):
        
        '''
            Return a matrix of pre-trained embeddings
        '''
        if not os.path.isfile(pretrained_emb):    
            print "Extracting %s -> %s" % (emb_path, pretrained_emb)  

            with open(emb_path) as fid:
                # Get emb size
                _, emb_size = fid.readline().split()
                # Get embeddings for all words in vocabulary
                E = np.zeros((int(emb_size), self.voc_size))   
                for line in fid.readlines():
                    items = line.split()
                    wrd   = items[0]
                    if wrd in self.wrd2idx:
                        E[:, self.wrd2idx[wrd]] = np.array(items[1:]).astype(float)
            #cache the matrix
            with open(pretrained_emb, 'w') as fid:
                cPickle.dump(E, fid, cPickle.HIGHEST_PROTOCOL)
        else:
            print "Loading %s" % pretrained_emb
            with open(pretrained_emb, 'r') as fid:
                E = cPickle.load(fid)

        return E