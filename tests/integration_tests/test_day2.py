from __future__ import division
import numpy as np
import pytest

import lxmls.readers.pos_corpus as pcc
import lxmls.readers.simple_sequence as ssr
import lxmls.sequences.confusion_matrix as cm
import lxmls.sequences.hmm as hmmc

tolerance = 1e-5
np.random.seed(4242)


@pytest.fixture(scope='module')
def simple():
    return ssr.SimpleSequence()


@pytest.fixture(scope='module')
def hmm(simple):
    return hmmc.HMM(simple.x_dict, simple.y_dict)


def test_exercise_1(simple):
    assert str(simple.train) == '[walk/rainy walk/sunny shop/sunny clean/sunny , walk/rainy walk/rainy shop/rainy clean/sunny , walk/sunny shop/sunny shop/sunny clean/sunny ]'
    assert str(simple.test) == '[walk/rainy walk/sunny shop/sunny clean/sunny , clean/sunny walk/sunny tennis/sunny walk/sunny ]'


def test_exercise_2(hmm, simple):
    hmm.train_supervised(simple.train)
    np.testing.utils.assert_almost_equal(hmm.initial_probs, np.array([2/3, 1/3]))
    np.testing.utils.assert_almost_equal(hmm.transition_probs, np.array([[1/2, 0.], [1/2, 5/8]]))
    np.testing.utils.assert_almost_equal(hmm.final_probs, np.array([0., 3/8]))
    np.testing.utils.assert_almost_equal(hmm.emission_probs, np.array([[0.75, 0.25], [0.25, 0.375], [0., 0.375], [0., 0.]]))


def test_exercise_3(hmm, simple):
    initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
    np.testing.utils.assert_almost_equal(initial_scores, np.array([-0.40546511, -1.09861229]))
    np.testing.utils.assert_almost_equal(transition_scores, np.array([[[-0.69314718, -np.inf],
                                                                       [-0.69314718, - 0.47000363]],
                                                                      [[-0.69314718, -np.inf],
                                                                       [-0.69314718, - 0.47000363]],
                                                                      [[-0.69314718, -np.inf],
                                                                       [-0.69314718, - 0.47000363]]]))
    np.testing.utils.assert_almost_equal(final_scores, np.array([-np.inf, -0.98082925]))
    np.testing.utils.assert_almost_equal(emission_scores, np.array([[-0.28768207, -1.38629436],
                                                                    [-0.28768207, -1.38629436],
                                                                    [-1.38629436, -0.98082925],
                                                                    [-np.inf, -0.98082925]]))


def test_exercise_4():
    a = np.random.rand(10)
    # print np.log(sum(np.exp(a)))
    # print np.log(sum(np.exp(10*a)))
    # print np.log(sum(np.exp(100*a)))
    # print np.log(sum(np.exp(1000*a)))

    from lxmls.sequences.log_domain import logsum

    assert abs(logsum(a) - 2.93067606639) < tolerance
    assert abs(logsum(10*a) - 10.185424506019405) < tolerance
    assert abs(logsum(100*a) - 94.110343934936424) < tolerance
    assert abs(logsum(1000*a) - 940.41457951845075) < tolerance


def test_exercise_5(hmm, simple):
    initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
    log_likelihood, forward = hmm.decoder.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
    assert abs(log_likelihood - -5.06823232601) < tolerance

    log_likelihood, backward = hmm.decoder.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
    assert abs(log_likelihood - -5.06823232601) < tolerance


def test_exercise_6(hmm, simple):
    initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
    state_posteriors, _, _ = hmm.compute_posteriors(initial_scores,
                                                    transition_scores,
                                                    final_scores,
                                                    emission_scores)
    np.testing.utils.assert_almost_equal(state_posteriors, np.array([[0.95738152, 0.04261848],
                                                                     [0.75281282, 0.24718718],
                                                                     [0.26184794, 0.73815206],
                                                                     [0., 1.]]))


def test_exercise_7(hmm, simple):
    y_pred = hmm.posterior_decode(simple.test.seq_list[0])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['rainy', 'rainy', 'sunny', 'sunny']

    y_pred = hmm.posterior_decode(simple.test.seq_list[1])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['rainy', 'rainy', 'rainy', 'rainy']

    hmm.train_supervised(simple.train, smoothing=0.1)

    y_pred = hmm.posterior_decode(simple.test.seq_list[0])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['rainy', 'rainy', 'sunny', 'sunny']

    y_pred = hmm.posterior_decode(simple.test.seq_list[1])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['sunny', 'sunny', 'sunny', 'sunny']


def test_exercise_8(hmm, simple):
    y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['rainy', 'rainy', 'sunny', 'sunny']
    assert abs(score - -6.02050124698) < tolerance

    y_pred, score = hmm.viterbi_decode(simple.test.seq_list[1])
    assert [y_pred.sequence_list.y_dict.get_label_name(yi) for yi in y_pred.y] == ['sunny', 'sunny', 'sunny', 'sunny']
    assert abs(score - -11.713974074) < tolerance


@pytest.fixture(scope='module')
def corpus_and_sequences():
    corpus = pcc.PostagCorpus()
    train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=15, max_nr_sent=1000)
    dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=15, max_nr_sent=1000)
    test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=15, max_nr_sent=1000)
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


@pytest.fixture(scope='module')
def hmm_with_corpus(corpus):
    return hmmc.HMM(corpus.word_dict, corpus.tag_dict)


def test_exercise_9(hmm_with_corpus, train_seq, dev_seq, test_seq):
    hmm_with_corpus.train_supervised(train_seq)

    viterbi_pred_train = hmm_with_corpus.viterbi_decode_corpus(train_seq)
    posterior_pred_train = hmm_with_corpus.posterior_decode_corpus(train_seq)
    eval_viterbi_train = hmm_with_corpus.evaluate_corpus(train_seq, viterbi_pred_train)
    eval_posterior_train = hmm_with_corpus.evaluate_corpus(train_seq, posterior_pred_train)
    assert abs(eval_posterior_train - 0.9848682232688646) < tolerance
    assert abs(eval_viterbi_train - 0.9846678023850085) < tolerance

    viterbi_pred_test = hmm_with_corpus.viterbi_decode_corpus(test_seq)
    posterior_pred_test = hmm_with_corpus.posterior_decode_corpus(test_seq)
    eval_viterbi_test = hmm_with_corpus.evaluate_corpus(test_seq, viterbi_pred_test)
    eval_posterior_test = hmm_with_corpus.evaluate_corpus(test_seq, posterior_pred_test)
    assert abs(eval_posterior_test - 0.3497722321251733) < tolerance
    assert abs(eval_viterbi_test - 0.5088136264606853) < tolerance

    best_smothing = hmm_with_corpus.pick_best_smoothing(train_seq, dev_seq, [10, 1, 0.1, 0])

    hmm_with_corpus.train_supervised(train_seq, smoothing=best_smothing)
    viterbi_pred_test = hmm_with_corpus.viterbi_decode_corpus(test_seq)
    posterior_pred_test = hmm_with_corpus.posterior_decode_corpus(test_seq)
    eval_viterbi_test = hmm_with_corpus.evaluate_corpus(test_seq, viterbi_pred_test)
    eval_posterior_test = hmm_with_corpus.evaluate_corpus(test_seq, posterior_pred_test)
    assert abs(best_smothing - 0.100) < tolerance
    assert abs(eval_posterior_test - 0.8367993662111309) < tolerance
    assert abs(eval_viterbi_test - 0.8265002970885323) < tolerance


def test_exercise_10(hmm_with_corpus, train_seq, test_seq):
    np.random.seed(4242)

    hmm_with_corpus.train_EM(train_seq, 0.1, 20, evaluate=True)
    viterbi_pred_test = hmm_with_corpus.viterbi_decode_corpus(test_seq)
    posterior_pred_test = hmm_with_corpus.posterior_decode_corpus(test_seq)
    eval_viterbi_test = hmm_with_corpus.evaluate_corpus(test_seq, viterbi_pred_test)
    eval_posterior_test = hmm_with_corpus.evaluate_corpus(test_seq, posterior_pred_test)

    assert abs(eval_posterior_test - 0.1437908496732026) < tolerance
    assert abs(eval_viterbi_test - 0.1437908496732026) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
