from __future__ import division

import numpy as np
import pytest

import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.crf_online as crfo
import lxmls.sequences.extended_feature as exfc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.structured_perceptron as spc

tolerance = 1e-5
np.random.seed(4242)


@pytest.fixture(scope='module')
def corpus_and_sequences():
    corpus = pcc.PostagCorpus()
    train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=10, max_nr_sent=1000)
    dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=10, max_nr_sent=1000)
    test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=10, max_nr_sent=1000)
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
    np.random.seed(4242)

    feature_mapper = idfc.IDFeatures(train_seq)
    feature_mapper.build_features()

    crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
    crf_online.num_epochs = 2
    crf_online.train_supervised(train_seq)

    pred_train = crf_online.viterbi_decode_corpus(train_seq)
    eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
    assert abs(eval_train - 0.8635761589403973) < tolerance  # for 2 epochs
    # assert abs(eval_train - 0.879028697571744) < tolerance  # for 3 epochs

    pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
    eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
    assert abs(eval_dev - 0.8083228247162674) < tolerance  # for 2 epochs
    # assert abs(eval_dev - 0.8184110970996217) < tolerance  # for 3 epochs

    pred_test = crf_online.viterbi_decode_corpus(test_seq)
    eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_test - 0.7967599410898379) < tolerance  # for 2 epochs
    # assert abs(eval_test - 0.804860088365) < tolerance  # for 3 epochs


def test_crf_extended_features(corpus, train_seq, dev_seq, test_seq):
    np.random.seed(4242)

    feature_mapper = exfc.ExtendedFeatures(train_seq)
    feature_mapper.build_features()

    crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
    crf_online.num_epochs = 2
    crf_online.train_supervised(train_seq)

    pred_train = crf_online.viterbi_decode_corpus(train_seq)
    eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
    assert abs(eval_train - 0.9361295069904342) < tolerance  # for 2 epochs
    # assert abs(eval_train - 0.9446651949963208) < tolerance  # for 3 epochs

    pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
    eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
    assert abs(eval_dev - 0.8827238335435057) < tolerance  # for 2 epochs
    # assert abs(eval_dev - 0.890290037831) < tolerance  # for 3 epochs

    pred_test = crf_online.viterbi_decode_corpus(test_seq)
    eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_test - 0.8637702503681886) < tolerance  # for 2 epochs
    # assert abs(eval_test - 0.865979381443) < tolerance  # for 3 epochs


def test_perceptron_id_features(corpus, train_seq, dev_seq, test_seq):
    np.random.seed(4242)

    feature_mapper = idfc.IDFeatures(train_seq)
    feature_mapper.build_features()

    sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
    sp.num_epochs = 3
    sp.train_supervised(train_seq)

    pred_train = sp.viterbi_decode_corpus(train_seq)
    pred_dev = sp.viterbi_decode_corpus(dev_seq)
    pred_test = sp.viterbi_decode_corpus(test_seq)
    eval_train = sp.evaluate_corpus(train_seq, pred_train)
    eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
    eval_test = sp.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_train - 0.9358351729212656) < tolerance
    assert abs(eval_dev - 0.839848675914) < tolerance
    assert abs(eval_test - 0.833578792342) < tolerance


def test_perceptron_extended_features(corpus, train_seq, dev_seq, test_seq):
    np.random.seed(4242)

    feature_mapper = exfc.ExtendedFeatures(train_seq)
    feature_mapper.build_features()

    sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
    sp.num_epochs = 3
    sp.train_supervised(train_seq)

    pred_train = sp.viterbi_decode_corpus(train_seq)
    pred_dev = sp.viterbi_decode_corpus(dev_seq)
    pred_test = sp.viterbi_decode_corpus(test_seq)
    eval_train = sp.evaluate_corpus(train_seq, pred_train)
    eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
    eval_test = sp.evaluate_corpus(test_seq, pred_test)
    assert abs(eval_train - 0.9346578366445916) < tolerance
    assert abs(eval_dev - 0.868852459016) < tolerance
    assert abs(eval_test - 0.865243004418) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
