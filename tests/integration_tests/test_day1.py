from __future__ import print_function

import numpy as np
import pytest

import lxmls.classifiers.gaussian_naive_bayes as gnbc
import lxmls.classifiers.max_ent_batch as mebc
import lxmls.classifiers.max_ent_online as meoc
import lxmls.classifiers.mira as mirac
import lxmls.classifiers.perceptron as percc
import lxmls.classifiers.svm as svmc
import lxmls.readers.sentiment_reader as srs
import lxmls.readers.simple_data_set as sds
from lxmls.classifiers import multinomial_naive_bayes

tolerance = 1e-5


@pytest.fixture(scope='module')
def sd():
    np.random.seed(4242)
    return sds.SimpleDataSet(nr_examples=100,
                             g1=[[-1, -1], 1],
                             g2=[[1, 1], 1],
                             balance=0.5,
                             split=[0.5, 0, 0.5])


# Exercise 3.1: run all classifiers on 2D data

def test_naive_bayes_simple_dataset(sd):
    np.random.seed(4242)

    gnb = gnbc.GaussianNaiveBayes()
    params_nb_sd = gnb.train(sd.train_X, sd.train_y)
    y_pred_train = gnb.test(sd.train_X, params_nb_sd)
    acc_train = gnb.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.88) < tolerance

    y_pred_test = gnb.test(sd.test_X, params_nb_sd)
    acc_test = gnb.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.90) < tolerance


def test_perceptron_simple_dataset(sd):
    np.random.seed(4242)

    perc = percc.Perceptron(nr_epochs=3)
    params_perc_sd = perc.train(sd.train_X, sd.train_y)
    y_pred_train = perc.test(sd.train_X, params_perc_sd)
    acc_train = perc.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.9) < tolerance

    y_pred_test = perc.test(sd.test_X, params_perc_sd)
    acc_test = perc.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.84) < tolerance


def test_mira_simple_dataset(sd):
    np.random.seed(4242)

    mira = mirac.Mira(nr_rounds=3)
    params_mira_sd = mira.train(sd.train_X, sd.train_y)
    y_pred_train = mira.test(sd.train_X, params_mira_sd)
    acc_train = mira.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.88) < tolerance

    y_pred_test = mira.test(sd.test_X, params_mira_sd)
    acc_test = mira.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.86) < tolerance


def test_maxent_batch_simple_dataset(sd):
    np.random.seed(4242)

    me_lbfgs = mebc.MaxEntBatch()
    params_meb_sd = me_lbfgs.train(sd.train_X, sd.train_y)
    y_pred_train = me_lbfgs.test(sd.train_X, params_meb_sd)
    acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.90) < tolerance

    y_pred_test = me_lbfgs.test(sd.test_X, params_meb_sd)
    acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.92) < tolerance


def test_maxent_online_simple_dataset(sd):
    np.random.seed(4242)

    me_sgd = meoc.MaxEntOnline(nr_epochs=3)
    params_meo_sd = me_sgd.train(sd.train_X, sd.train_y)
    y_pred_train = me_sgd.test(sd.train_X, params_meo_sd)
    acc_train = me_sgd.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.86) < tolerance

    y_pred_test = me_sgd.test(sd.test_X, params_meo_sd)
    acc_test = me_sgd.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.92) < tolerance


def test_svm_online_simple_dataset(sd):
    np.random.seed(4242)

    svm = svmc.SVM(nr_epochs=3)
    params_svm_sd = svm.train(sd.train_X, sd.train_y)
    y_pred_train = svm.test(sd.train_X, params_svm_sd)
    acc_train = svm.evaluate(sd.train_y, y_pred_train)
    assert abs(acc_train - 0.88) < tolerance

    y_pred_test = svm.test(sd.test_X, params_svm_sd)
    acc_test = svm.evaluate(sd.test_y, y_pred_test)
    assert abs(acc_test - 0.92) < tolerance


@pytest.fixture(scope='module')
def scr():
    # Read the book review data
    np.random.seed(4242)
    return srs.SentimentCorpus("books")


def test_naive_bayes_amazon_sentiment(scr):
    np.random.seed(4242)

    mnb = multinomial_naive_bayes.MultinomialNaiveBayes()
    params_nb_sc = mnb.train(scr.train_X, scr.train_y)
    y_pred_train = mnb.test(scr.train_X, params_nb_sc)
    acc_train = mnb.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.987500) < tolerance

    y_pred_test = mnb.test(scr.test_X, params_nb_sc)
    acc_test = mnb.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.635000) < tolerance
    # FIXME: depends on the seed: You should get 0.656250 on the train set and 0.622500 on the test set


def test_perceptron_amazon_sentiment(scr):
    np.random.seed(4242)

    perc = percc.Perceptron(nr_epochs=3)
    params_perc_sc = perc.train(scr.train_X, scr.train_y)
    y_pred_train = perc.test(scr.train_X, params_perc_sc)
    acc_train = perc.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.966875) < tolerance

    y_pred_test = perc.test(scr.test_X, params_perc_sc)
    acc_test = perc.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.805) < tolerance


def test_mira_amazon_sentiment(scr):
    np.random.seed(4242)

    mira = mirac.Mira(nr_rounds=1)
    params_mira_sc = mira.train(scr.train_X, scr.train_y)
    y_pred_train = mira.test(scr.train_X, params_mira_sc)
    acc_train = mira.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.50) < tolerance

    y_pred_test = mira.test(scr.test_X, params_mira_sc)
    acc_test = mira.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.50) < tolerance


def test_maxent_batch_amazon_sentiment(scr):
    np.random.seed(4242)

    me_lbfgs = mebc.MaxEntBatch()
    params_meb_sc = me_lbfgs.train(scr.train_X, scr.train_y)
    y_pred_train = me_lbfgs.test(scr.train_X, params_meb_sc)
    acc_train = me_lbfgs.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.858125) < tolerance

    y_pred_test = me_lbfgs.test(scr.test_X, params_meb_sc)
    acc_test = me_lbfgs.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.790000) < tolerance


def test_maxent_online_amazon_sentiment(scr):
    np.random.seed(4242)

    me_sgd = meoc.MaxEntOnline(nr_epochs=3)
    params_meo_sc = me_sgd.train(scr.train_X, scr.train_y)
    y_pred_train = me_sgd.test(scr.train_X, params_meo_sc)
    acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.85125) < tolerance

    y_pred_test = me_sgd.test(scr.test_X, params_meo_sc)
    acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.7775) < tolerance


def test_svm_online_amazon_sentiment(scr):
    np.random.seed(4242)

    svm = svmc.SVM(nr_epochs=3)
    params_svm_sc = svm.train(scr.train_X, scr.train_y)
    y_pred_train = svm.test(scr.train_X, params_svm_sc)
    acc_train = svm.evaluate(scr.train_y, y_pred_train)
    assert abs(acc_train - 0.845625) < tolerance

    y_pred_test = svm.test(scr.test_X, params_svm_sc)
    acc_test = svm.evaluate(scr.test_y, y_pred_test)
    assert abs(acc_test - 0.7725) < tolerance


if __name__ == '__main__':
    pytest.main([__file__])
