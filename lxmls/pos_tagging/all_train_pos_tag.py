import os
import sys
import codecs

from lxmls import data
from sequences.sequence import *
from sequences.sequence_list import *
import readers.pos_corpus as pcc
import readers.brown_pos_corpus as bpc
import sequences.extended_feature as exfc
import sequences.structured_perceptron as spc
import sequences.confusion_matrix as bcm

MAX_SENT_SIZE = 1000
MAX_NR_SENTENCES = 100000
MODEL_DIR = "/Users/graca/Projects/swm_src/feeds/models/all_data_postag/"


def build_corpus_features():
    corpus = pcc.PostagCorpus()
    train_seq = corpus.read_sequence_list_conll(data.find('train-02-21.conll'),
        max_sent_len=MAX_SENT_SIZE,
        max_nr_sent=MAX_NR_SENTENCES)
    corpus.add_sequence_list(train_seq)
    dev_seq = corpus.read_sequence_list_conll(data.find('dev-22.conll'))
    corpus.add_sequence_list(dev_seq)
    categories = [
        'adventure',
        'belles_lettres',
        'editorial',
        'fiction',
        'government',
        'hobbies',
        'humor',
        'learned',
        'lore',
        'mystery',
        'news',
        'religion',
        'reviews',
        'romance']
    for cat in categories:
        brown_seq = corpus.read_sequence_list_brown(categories=cat)
        corpus.add_sequence_list(brown_seq)
    features = exfc.ExtendedFeatures(corpus)
    features.build_features()
    corpus.save_corpus(MODEL_DIR)
    features.save_features(MODEL_DIR+"features.txt")
    return corpus, features


def train_pos(corpus, features):
    model = spc.StructuredPercetron(corpus, features)
    model.nr_rounds = 10
    model.train_supervised(corpus.sequence_list.seq_list)
    model.save_model(MODEL_DIR)
    return model


def eval_model(corpus, features, model):
    test_seq = corpus.read_sequence_list_conll(data.find('test-23.conll'))
    pred_test = model.viterbi_decode_corpus_log(test_seq.seq_list)
    eval_test = model.evaluate_corpus(test_seq.seq_list, pred_test)
    print("Accuracy on wsj test %f" % eval_test)


def eval_brown(corpus, features, model):
    categories = ['science_fiction']
    for cat in categories:
        brown_seq = corpus.read_sequence_list_brown(categories=cat)
        brown_pred = model.viterbi_decode_corpus_log(brown_seq.seq_list)
        brown_eval = model.evaluate_corpus(brown_seq.seq_list, brown_pred)
        print("Accuracy on Brown cat %s: %f" % (cat, brown_eval))


def load_model():
    corpus = pcc.PostagCorpus()
    corpus.load_corpus(MODEL_DIR)
    features = exfc.ExtendedFeatures(corpus)
    features.load_features(MODEL_DIR+"features.txt", corpus)
    model = spc.StructuredPercetron(corpus, features)
    model.load_model(MODEL_DIR)
    return corpus, features, model


def main():
    print("Building corpus")
    corpus, features = build_corpus_features()
    print("Training model")
    model = train_pos(corpus, features)
    print("Testing on wsj")
    eval_model(corpus, features, model)
    print("Testing on brown")
    eval_brown(corpus, features, model)
    # print "Loading models"
    # corpus,features,model = load_model()
    # print "Testing on wsj"
    # eval_model(corpus,features,model)
    # print "Testing on brown"
    # eval_brown(corpus,features,model)
