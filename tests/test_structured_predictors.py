import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)

import numpy as np

import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.crf_online as crfo
import lxmls.sequences.extended_feature as exfc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.structured_perceptron as spc
from lxmls import data

tolerance = 1e-5

@pytest.fixture(scope='module')
def corpus_and_sequences():
    corpus = pcc.PostagCorpus()
    train_seq = corpus.read_sequence_list_conll(data.find('train-02-21.conll'), max_sent_len=10, max_nr_sent=1000)
    dev_seq = corpus.read_sequence_list_conll(data.find('dev-22.conll'), max_sent_len=10, max_nr_sent=1000)
    test_seq = corpus.read_sequence_list_conll(data.find('test-23.conll'), max_sent_len=10, max_nr_sent=1000)
    return corpus, train_seq, dev_seq, test_seq


@pytest.fixture(scope='module')
def corpus(corpus_and_sequences):
    return corpus_and_sequences[0]


@pytest.fixture(scope='module')
def train_seq(corpus_and_sequences):
    return corpus_and_sequences[1]


@pytest.fixture(scope='module')
def dev_seq(corpus_and_sequences):
    return corpus_and_sequences[2]


@pytest.fixture(scope='module')
def test_seq(corpus_and_sequences):
    return corpus_and_sequences[3]


def test_crf_id_features(corpus, train_seq, dev_seq, test_seq):

    feature_mapper = idfc.IDFeatures(train_seq)
    feature_mapper.build_features()

    crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
    crf_online.num_epochs = 1
    crf_online.train_supervised(train_seq)

    pred_train = crf_online.viterbi_decode_corpus(train_seq)
    eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
    assert abs(eval_train - 0.8394407652685798) < tolerance

    pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
    eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
    assert abs(eval_dev - 0.7957124842370744) < tolerance

    pred_test = crf_online.viterbi_decode_corpus(test_seq)
    eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_test - 0.7849779086892489) < tolerance


def test_crf_extended_features(corpus, train_seq, dev_seq, test_seq):

    feature_mapper = exfc.ExtendedFeatures(train_seq)
    feature_mapper.build_features()

    crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
    crf_online.num_epochs = 1
    crf_online.train_supervised(train_seq)

    pred_train = crf_online.viterbi_decode_corpus(train_seq)
    eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
    assert abs(eval_train - 0.9225901398086829) < tolerance

    pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
    eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
    assert abs(eval_dev - 0.8738965952080706) < tolerance

    pred_test = crf_online.viterbi_decode_corpus(test_seq)
    eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_test - 0.8630338733431517) < tolerance


def test_perceptron_id_features(corpus, train_seq, dev_seq, test_seq):

    feature_mapper = idfc.IDFeatures(train_seq)
    feature_mapper.build_features()

    sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
    sp.num_epochs = 1
    sp.train_supervised(train_seq)

    pred_train = sp.viterbi_decode_corpus(train_seq)
    pred_dev = sp.viterbi_decode_corpus(dev_seq)
    pred_test = sp.viterbi_decode_corpus(test_seq)
    eval_train = sp.evaluate_corpus(train_seq, pred_train)
    eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
    eval_test = sp.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_train - 0.7980868285504047) < tolerance
    assert abs(eval_dev - 0.7641866330390921) < tolerance
    assert abs(eval_test - 0.7187039764359352) < tolerance


def test_perceptron_extended_features(corpus, train_seq, dev_seq, test_seq):

    feature_mapper = exfc.ExtendedFeatures(train_seq)
    feature_mapper.build_features()

    sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
    sp.num_epochs = 1
    sp.train_supervised(train_seq)

    pred_train = sp.viterbi_decode_corpus(train_seq)
    pred_dev = sp.viterbi_decode_corpus(dev_seq)
    pred_test = sp.viterbi_decode_corpus(test_seq)
    eval_train = sp.evaluate_corpus(train_seq, pred_train)
    eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
    eval_test = sp.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_train - 0.8679911699779249) < tolerance
    assert abs(eval_dev - 0.8310214375788146) < tolerance
    assert abs(eval_test - 0.8181148748159057) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
