import pytest

from numpy import allclose
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.multinomial_naive_bayes as mnbb
import warnings

tolerance = 1e-5

@pytest.fixture(scope='module')
def sentiment_corpus():
    return srs.SentimentCorpus("books")

def test_naive_bayes(sentiment_corpus):
    mnb = mnbb.MultinomialNaiveBayes()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # In this exercise, python should yield the following warning:
        # RuntimeWarning: divide by zero encountered in log
        params_nb_sc = mnb.train(sentiment_corpus.train_X,sentiment_corpus.train_y)
        # TODO: make a test to check if the warning was issued
    
    y_pred_train = mnb.test(sentiment_corpus.train_X,params_nb_sc)
    acc_train = mnb.evaluate(sentiment_corpus.train_y, y_pred_train)
    assert allclose(acc_train, 0.987500, tolerance)

    y_pred_test = mnb.test(sentiment_corpus.test_X,params_nb_sc)
    acc_test = mnb.evaluate(sentiment_corpus.test_y, y_pred_test)
    assert allclose(acc_test, 0.635000, tolerance)

if __name__ == '__main__':
    pytest.main([__file__])


