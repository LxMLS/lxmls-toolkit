from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pytest
import theano
import theano.tensor as T

import lxmls.deep_learning.embeddings as emb
import lxmls.deep_learning.rnn as rnns
import lxmls.readers.pos_corpus as pcc

tolerance = 1e-5
np.random.seed(4242)

SEED = 1234  # Random seed to initialize weigths
emb_size = 50  # Size of word embeddings
hidden_size = 20  # size of hidden layer
lrate = 0.5
n_iter = 3


# Function computing accuracy for a sequence of sentences
def accuracy(seq, err_sum):
    err = 0
    N = 0
    for n, seq in enumerate(seq):
        err += err_sum(seq.x, seq.y)
        N += seq.y.shape[0]
    return 100 * (1 - err/N)


@pytest.fixture(scope='module')
def corpus_and_sequences():
    corpus = pcc.PostagCorpus()
    train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=15, max_nr_sent=1000)
    dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=15, max_nr_sent=1000)
    test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=15, max_nr_sent=1000)
    # Redo indices so that they are consecutive. Also cast all data to numpy arrays
    # of int32 for compatibility with GPUs and theano and add reverse index
    train_seq, test_seq, dev_seq = pcc.compacify(train_seq, test_seq, dev_seq, theano=True)
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
def embeddings(train_seq):
    if not os.path.isfile('data/senna_50'):
        emb.download_embeddings('senna_50', 'data/senna_50')
    return emb.extract_embeddings('data/senna_50', train_seq.x_dict)


def test_exercise_1(train_seq):
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)

    np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
    x0 = train_seq[0].x
    y0 = train_seq[0].y
    p_y, y_rnn, h, z1, x = np_rnn.forward(x0, all_outputs=True)
    expected_p_y = np.array([[0.16200125, 0.24483233, 0.24521576, 0.24521403, 0.24521506],
                             [0.03020501, 0.01080573, 0.01062576, 0.01062537, 0.01062548],
                             [0.11620671, 0.12908527, 0.12960888, 0.12961018, 0.12960979],
                             [0.06182014, 0.04062052, 0.04045927, 0.04045879, 0.04045896],
                             [0.13363292, 0.17044615, 0.1703328, 0.17033331, 0.17033286],
                             [0.06001475, 0.0379502, 0.03787886, 0.03787929, 0.03787892],
                             [0.09645537, 0.09236851, 0.09231392, 0.09231365, 0.09231372],
                             [0.12219898, 0.14190367, 0.1427655, 0.14276907, 0.14276764],
                             [0.05585568, 0.03369287, 0.03326684, 0.03326566, 0.03326617],
                             [0.06942549, 0.0505977, 0.05053546, 0.05053532, 0.05053553],
                             [0.04170364, 0.0194946, 0.01909294, 0.01909194, 0.01909226],
                             [0.05048007, 0.02820244, 0.027904, 0.02790338, 0.02790361]])
    assert np.allclose(p_y, expected_p_y, rtol=tolerance)

    numpy_rnn_gradients = np_rnn.grads(x0, y0)
    gradients_means = [np.mean(grad) for grad in numpy_rnn_gradients]
    expected_means = [-5.2579723782223409e-06, -1.4733517199834271e-05, -4.213560027964384e-06, -3.7007434154171883e-17]
    assert np.allclose(gradients_means, expected_means, rtol=tolerance)


def test_exercise_2():
    np.random.seed(SEED)
    theano.config.optimizer = 'None'

    def square(x):
        return x**2

    def np_square_n_steps(nr_steps):
        out = []
        for n in np.arange(nr_steps):
            out.append(square(n))
        return np.array(out)

    # Theano
    nr_steps = T.lscalar('nr_steps')
    h, _ = theano.scan(fn=square, sequences=T.arange(nr_steps))
    th_square_n_steps = theano.function([nr_steps], h)

    assert np.allclose(np_square_n_steps(10), [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    assert np.allclose(th_square_n_steps(10), [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])

    nr_states = 3
    nr_steps = 5

    # Transition matrix
    A = np.abs(np.random.randn(nr_states, nr_states))
    A = A / A.sum(0, keepdims=True)
    # Initial state
    s0 = np.zeros(nr_states)
    s0[0] = 1

    # Numpy version
    def np_markov_step(s_tm1):
        s_t = np.dot(s_tm1, A.T)
        return s_t

    def np_markov_chain(nr_steps, A, s0):
        # Pre-allocate space
        s = np.zeros((nr_steps + 1, nr_states))
        s[0, :] = s0
        for t in np.arange(nr_steps):
            s[t + 1, :] = np_markov_step(s[t, :])
        return s

    output = np_markov_chain(nr_steps, A, s0)
    expected_output = np.array([[1., 0., 0.],
                                [0.28681767, 0.1902151, 0.52296723],
                                [0.49197478, 0.30699777, 0.20102745],
                                [0.40791406, 0.25675885, 0.3353271],
                                [0.44270576, 0.27757548, 0.27971876],
                                [0.42830249, 0.26895746, 0.30274005]])
    assert np.allclose(output, expected_output, rtol=tolerance)

    # Theano version
    # Store variables as shared variables
    th_A = theano.shared(A, name='A', borrow=True)
    th_s0 = theano.shared(s0, name='s0', borrow=True)
    # Symbolic variable for the number of steps
    th_nr_steps = T.lscalar('nr_steps')

    def th_markov_step(s_tm1):
        s_t = T.dot(s_tm1, th_A.T)
        # Remember to name variables
        s_t.name = 's_t'
        return s_t

    s, _ = theano.scan(th_markov_step,
                       outputs_info=[dict(initial=th_s0)],
                       n_steps=th_nr_steps)
    th_markov_chain = theano.function([th_nr_steps], T.concatenate((th_s0[None, :], s), 0))

    output = th_markov_chain(nr_steps)
    expected_output = np.array([[1., 0., 0.],
                                [0.28681767, 0.1902151, 0.52296723],
                                [0.49197478, 0.30699777, 0.20102745],
                                [0.40791406, 0.25675885, 0.3353271],
                                [0.44270576, 0.27757548, 0.27971876],
                                [0.42830249, 0.26895746, 0.30274005]])
    assert np.allclose(output, expected_output, rtol=tolerance)


def test_exercise_3(train_seq, dev_seq):
    np.random.seed(SEED)
    # Reused data
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)
    np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
    x0 = train_seq[0].x
    y0 = train_seq[0].y
    numpy_rnn_gradients = np_rnn.grads(x0, y0)

    # Begin test
    rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)

    x = T.ivector('x')
    th_forward = theano.function([x], rnn._forward(x).T)

    assert np.allclose(th_forward(x0), np_rnn.forward(x0))

    # Compile function returning the list of gradients
    x = T.ivector('x')  # Input words
    y = T.ivector('y')  # gold tags
    p_y = rnn._forward(x)
    cost = -T.mean(T.log(p_y)[T.arange(y.shape[0]), y])
    grads_fun = theano.function([x, y], [T.grad(cost, par) for par in rnn.param])

    # Compare numpy and theano gradients
    theano_rnn_gradients = grads_fun(x0, y0)
    for n in range(len(theano_rnn_gradients)):
        assert np.allclose(numpy_rnn_gradients[n], theano_rnn_gradients[n])

    rnn_prediction = theano.function([x], T.argmax(p_y, 1))
    predicted_tags = [train_seq.tag_dict[pred] for pred in rnn_prediction(train_seq[0].x)]
    expected_tags = [u'noun', u'noun', u'noun', u'noun', u'noun']
    assert predicted_tags == expected_tags

    # Get list of SGD batch update rule for each parameter
    updates = [(par, par - lrate * T.grad(cost, par)) for par in rnn.param]
    # compile
    rnn_batch_update = theano.function([x, y], cost, updates=updates)

    nr_words = sum([len(seq.x) for seq in train_seq])
    expected_training_values = [(2305.557593, 0.39763503), (1086.043786, 0.79196312), (399.323869, 0.96873434)]
    expected_dev_accuracies = [0.82513278, 0.91592344, 0.94167752]
    for i in range(n_iter):
        # Training
        cost = 0.
        errors = 0
        for n, seq in enumerate(train_seq):
            cost += rnn_batch_update(seq.x, seq.y)
            errors += sum(rnn_prediction(seq.x) != seq.y)
        acc_train = 1. - errors/nr_words
        assert abs(cost - expected_training_values[i][0]) < tolerance
        assert abs(acc_train - expected_training_values[i][1]) < tolerance

        # Evaluation
        errors = 0
        for n, seq in enumerate(dev_seq):
            errors += sum(rnn_prediction(seq.x) != seq.y)
        acc_dev = 1. - errors/nr_words
        assert abs(acc_dev - expected_dev_accuracies[i]) < tolerance


def test_exercise_4(train_seq, dev_seq, embeddings):
    np.random.seed(SEED)
    # Reused data
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)
    rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
    x = T.ivector('x')  # Input words
    y = T.ivector('y')  # gold tags

    # Set the embedding layer to the pre-trained values
    rnn.param[0].set_value(embeddings.astype(theano.config.floatX))

    lstm = rnns.LSTM(nr_words, emb_size, hidden_size, nr_tags, seed=SEED)
    lstm_prediction = theano.function([x], T.argmax(lstm._forward(x), 1))
    lstm_cost = -T.mean(T.log(lstm._forward(x))[T.arange(y.shape[0]), y])

    # Get list of SGD batch update rule for each parameter
    lstm_updates = [(par, par - lrate * T.grad(lstm_cost, par)) for par in lstm.param]
    lstm_batch_update = theano.function([x, y], lstm_cost, updates=lstm_updates)

    nr_words = sum([len(seq.x) for seq in train_seq])
    # With 3 iterations
    # epochs = n_iter
    # expected_training_values = [(2440.5075314676142, 0.286100811705), (2378.56841952, 0.322677623008), (2047.61953503, 0.418879647259)]
    # expected_dev_accuracies = [0.747870528109, 0.750576210041, 0.805892373985]
    epochs = 1
    expected_training_values = [(2440.5075314676142, 0.286100811705)]
    expected_dev_accuracies = [0.747870528109]
    for i in range(epochs):
        # Training
        cost = 0
        errors = 0
        for n, seq in enumerate(train_seq):
            cost += lstm_batch_update(seq.x, seq.y)
            errors += sum(lstm_prediction(seq.x) != seq.y)
        acc_train = 1. - errors/nr_words
        assert abs(cost - expected_training_values[i][0]) < tolerance
        assert abs(acc_train - expected_training_values[i][1]) < tolerance

        # Evaluation
        errors = 0
        for n, seq in enumerate(dev_seq):
            errors += sum(lstm_prediction(seq.x) != seq.y)
        acc_dev = 1. - errors/nr_words
        assert abs(acc_dev - expected_dev_accuracies[i]) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
