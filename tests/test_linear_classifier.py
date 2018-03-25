import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)
from numpy import allclose
import warnings

from lxmls.classifiers import multinomial_naive_bayes as mnbb
import lxmls.classifiers.gaussian_naive_bayes as gnbc
import lxmls.classifiers.max_ent_batch as mebc
import lxmls.classifiers.max_ent_online as meoc
import lxmls.classifiers.mira as mirac
import lxmls.classifiers.perceptron as percc
import lxmls.classifiers.svm as svmc
import lxmls.readers.sentiment_reader as srs
import lxmls.readers.simple_data_set as sds

tolerance = 1e-5

@pytest.fixture(scope='module')
def scr():
    return srs.SentimentCorpus("books")

# Exercise 1.1
def test_naive_bayes(scr):

    mnb = mnbb.MultinomialNaiveBayes()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # In this exercise, python should yield the following warning:
        # RuntimeWarning: divide by zero encountered in log
        params_nb_sc = mnb.train(scr.train_X,scr.train_y)
        # TODO: make a test to check if the warning was issued
    
    y_pred_train = mnb.test(scr.train_X,params_nb_sc)
    acc_train = mnb.evaluate(scr.train_y, y_pred_train)
    assert allclose(acc_train, 0.987500, tolerance)

    y_pred_test = mnb.test(scr.test_X,params_nb_sc)
    acc_test = mnb.evaluate(scr.test_y, y_pred_test)
    assert allclose(acc_test, 0.635000, tolerance)

@pytest.fixture(scope='module')
def sd():
    return sds.SimpleDataSet(
        nr_examples=100,
        g1=[[-1,-1],1], 
        g2=[[1,1],1], 
        balance=0.5,
        split=[0.5,0,0.5]
    )

# Exercise 1.2
def test_perceptron(sd):
    perc = percc.Perceptron()
    params_perc_sd = perc.train(sd.train_X,sd.train_y)

    y_pred_train = perc.test(sd.train_X,params_perc_sd)
    acc_train = perc.evaluate(sd.train_y, y_pred_train)
    assert allclose(acc_train, 0.960000, tolerance)

    y_pred_test = perc.test(sd.test_X,params_perc_sd)
    acc_test = perc.evaluate(sd.test_y, y_pred_test)
    assert allclose(acc_test, 0.960000, tolerance)

# Exercise 1.3
def test_mira(sd):
    mira = mirac.Mira()
    mira.regularizer = 1.0 # This is lambda
    params_mira_sd = mira.train(sd.train_X,sd.train_y)

    y_pred_train = mira.test(sd.train_X,params_mira_sd)
    acc_train = mira.evaluate(sd.train_y, y_pred_train)
    assert allclose(acc_train, 0.960000, tolerance)

    y_pred_test = mira.test(sd.test_X,params_mira_sd)
    acc_test = mira.evaluate(sd.test_y, y_pred_test)
    assert allclose(acc_test, 0.960000, tolerance)

# Exercise 1.4
def test_maxent_batch(sd, scr):
    me_lbfgs = mebc.MaxEntBatch()

    params_meb_sd = me_lbfgs.train(sd.train_X, sd.train_y)
    y_pred_train = me_lbfgs.test(sd.train_X, params_meb_sd)
    acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
    assert allclose(acc_train, 0.980000, tolerance)

    y_pred_test = me_lbfgs.test(sd.test_X, params_meb_sd)
    acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
    assert allclose(acc_test, 0.960000, tolerance)

    params_meb_sc = me_lbfgs.train(scr.train_X,scr.train_y)
    y_pred_train = me_lbfgs.test(scr.train_X,params_meb_sc)
    acc_train = me_lbfgs.evaluate(scr.train_y, y_pred_train)
    assert allclose(acc_train, 0.858125, tolerance)

    y_pred_test = me_lbfgs.test(scr.test_X,params_meb_sc)
    acc_test = me_lbfgs.evaluate(scr.test_y, y_pred_test)
    assert allclose(acc_test, 0.790000, tolerance)


def test_maxent_online(scr):
    me_sgd = meoc.MaxEntOnline()
    me_sgd.regularizer = 1.0
    params_meo_sc = me_sgd.train(scr.train_X,scr.train_y)

    y_pred_train = me_sgd.test(scr.train_X,params_meo_sc)
    acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
    assert allclose(acc_train, 0.860000, tolerance)

    y_pred_test = me_sgd.test(scr.test_X,params_meo_sc)
    acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)
    assert allclose(acc_test, 0.795000, tolerance)   

# Exercise 1.5
def test_svm_online(sd, scr):
    svm = svmc.SVM()
    svm.regularizer = 1.0 # This is lambda
    params_svm_sd = svm.train(sd.train_X,sd.train_y)

    y_pred_train = svm.test(sd.train_X,params_svm_sd)
    acc_train = svm.evaluate(sd.train_y, y_pred_train)
    assert allclose(acc_train, 0.940000, tolerance)

    y_pred_test = svm.test(sd.test_X,params_svm_sd)
    acc_test = svm.evaluate(sd.test_y, y_pred_test)
    assert allclose(acc_test, 0.960000, tolerance)

    params_svm_sc = svm.train(scr.train_X,scr.train_y)

    y_pred_train = svm.test(scr.train_X,params_svm_sc)
    acc_train = svm.evaluate(scr.train_y, y_pred_train)
    assert allclose(acc_train, 0.87875, 0.01)

    y_pred_test = svm.test(scr.test_X,params_svm_sc)
    acc_test = svm.evaluate(scr.test_y, y_pred_test)
    # TODO: py2 gives 0.805, check the reason for the different value
    assert allclose(acc_test, 0.810000, 0.01)

if __name__ == '__main__':
    pytest.main([__file__])


