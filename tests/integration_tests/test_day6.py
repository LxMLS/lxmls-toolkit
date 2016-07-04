from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pytest
import theano
import theano.tensor as T

import lxmls.deep_learning.embeddings as emb
import lxmls.deep_learning.rnn as rnns
import lxmls.readers.pos_corpus as pcc

tolerance = 1e-5
seed = 4242

emb_size = 50  # Size of word embeddings
hidden_size = 20  # size of hidden layer
lrate = 0.5
n_iter = 1


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
    np.random.seed(seed)
    if not os.path.isfile('data/senna_50'):
        emb.download_embeddings('senna_50', 'data/senna_50')
    return emb.extract_embeddings('data/senna_50', train_seq.x_dict)


def test_exercise_1(train_seq):
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)

    np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=seed)
    x0 = train_seq[0].x
    y0 = train_seq[0].y
    p_y, y_rnn, h, z1, x = np_rnn.forward(x0, all_outputs=True)
    expected_p_y = np.array([[0.03468945, 0.01273397, 0.01260221, 0.01260213, 0.01260212],
                             [0.06979642, 0.04786876, 0.04771255, 0.04771253, 0.04771249],
                             [0.07680857, 0.05766041, 0.05738554, 0.05738532, 0.05738534],
                             [0.09176535, 0.0798503, 0.07895001, 0.07894913, 0.07894882],
                             [0.02758537, 0.00831099, 0.00821173, 0.00821167, 0.00821165],
                             [0.13609027, 0.1671459, 0.16810529, 0.16810628, 0.16810643],
                             [0.16687838, 0.24717632, 0.24817336, 0.24817411, 0.24817432],
                             [0.05536858, 0.03116249, 0.0309114, 0.03091119, 0.03091119],
                             [0.12557292, 0.14314541, 0.14311647, 0.1431164, 0.14311642],
                             [0.0408884, 0.01748585, 0.01720501, 0.01720476, 0.0172047],
                             [0.03478682, 0.01279824, 0.01265781, 0.01265771, 0.01265768],
                             [0.13976949, 0.17466137, 0.17496863, 0.17496879, 0.17496884]])

    assert np.allclose(p_y, expected_p_y, rtol=tolerance)

    numpy_rnn_gradients = np_rnn.grads(x0, y0)
    gradients_means = [np.mean(grad) for grad in numpy_rnn_gradients]
    expected_means = [1.0494697085525477e-05, 2.4443266955056405e-05, 5.7089383719915436e-05, 3.7007434154171883e-17]
    assert np.allclose(gradients_means, expected_means, rtol=tolerance)


def test_exercise_2():
    np.random.seed(seed)
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
                                [0.14517224, 0.15967512, 0.69515264],
                                [0.14832788, 0.60377597, 0.24789616],
                                [0.14584665, 0.52849836, 0.32565499],
                                [0.14627644, 0.5423297, 0.31139386],
                                [0.1461976, 0.53980273, 0.31399967]])

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
                                [0.14517224, 0.15967512, 0.69515264],
                                [0.14832788, 0.60377597, 0.24789616],
                                [0.14584665, 0.52849836, 0.32565499],
                                [0.14627644, 0.5423297, 0.31139386],
                                [0.1461976, 0.53980273, 0.31399967]])
    assert np.allclose(output, expected_output, rtol=tolerance)


def test_exercise_3(train_seq, dev_seq):
    np.random.seed(seed)
    # Reused data
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)
    np_rnn = rnns.NumpyRNN(nr_words, emb_size, hidden_size, nr_tags, seed=seed)
    x0 = train_seq[0].x
    y0 = train_seq[0].y
    numpy_rnn_gradients = np_rnn.grads(x0, y0)

    # Begin test
    rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=seed)

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
    expected_tags = [u'adp', u'adp', u'adp', u'adp', u'adp']
    assert predicted_tags == expected_tags

    # Get list of SGD batch update rule for each parameter
    updates = [(par, par - lrate * T.grad(cost, par)) for par in rnn.param]
    # compile
    rnn_batch_update = theano.function([x, y], cost, updates=updates)

    nr_words = sum([len(seq.x) for seq in train_seq])
    expected_training_values = [(2517.4383217341324, 0.34271971139392721)]
    expected_dev_accuracies = [0.7731235594748973]
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
    np.random.seed(seed)
    # Reused data
    nr_words = len(train_seq.x_dict)
    nr_tags = len(train_seq.y_dict)
    rnn = rnns.RNN(nr_words, emb_size, hidden_size, nr_tags, seed=seed)
    x = T.ivector('x')  # Input words
    y = T.ivector('y')  # gold tags

    # Set the embedding layer to the pre-trained values
    rnn.param[0].set_value(embeddings.astype(theano.config.floatX))

    lstm = rnns.LSTM(nr_words, emb_size, hidden_size, nr_tags, seed=seed)
    lstm_prediction = theano.function([x], T.argmax(lstm._forward(x), 1))
    lstm_cost = -T.mean(T.log(lstm._forward(x))[T.arange(y.shape[0]), y])

    # Get list of SGD batch update rule for each parameter
    lstm_updates = [(par, par - lrate * T.grad(lstm_cost, par)) for par in lstm.param]
    lstm_batch_update = theano.function([x, y], lstm_cost, updates=lstm_updates)

    nr_words = sum([len(seq.x) for seq in train_seq])
    expected_training_values = [(2460.9543686390084, 0.28640144303036374)]
    expected_dev_accuracies = [0.74957410562180582]
    cost_tolerance = 10  # FIXME: using a large tolerance because `emb.extract_embeddings` cannot be directly seeded.
    acc_tolerance = 1e-2  # FIXME: using a large tolerance because `emb.extract_embeddings` cannot be directly seeded.
    for i in range(n_iter):
        # Training
        cost = 0
        errors = 0
        for n, seq in enumerate(train_seq):
            cost += lstm_batch_update(seq.x, seq.y)
            errors += sum(lstm_prediction(seq.x) != seq.y)
        acc_train = 1. - errors/nr_words
        assert abs(cost - expected_training_values[i][0]) < cost_tolerance
        assert abs(acc_train - expected_training_values[i][1]) < acc_tolerance

        # Evaluation
        errors = 0
        for n, seq in enumerate(dev_seq):
            errors += sum(lstm_prediction(seq.x) != seq.y)
        acc_dev = 1. - errors/nr_words
        assert abs(acc_dev - expected_dev_accuracies[i]) < acc_tolerance


if __name__ == '__main__':
    pytest.main([__file__])
