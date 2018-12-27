import sys

sys.path.append("readers/")
sys.path.append("classifiers/")

import simple_data_set as sds

import linear_classifier as lcc
import naive_bayes as nbc
import perceptron as percc
import svm as svmc
import mira as mirac
import max_ent_batch as mec_batch
import max_ent_online as mec_online


def run_all_classifiers(dataset):
    fig, axis = dataset.plot_data()
    print("Naive Bayes")
    nb = nbc.NaiveBayes()
    params_nb = nb.train(dataset.train_X, dataset.train_y)
    print(params_nb.reshape(-1))
    predict = nb.test(dataset.train_X, params_nb)
    evaluation = nb.evaluate(predict, dataset.train_y)
    predict2 = nb.test(dataset.test_X, params_nb)
    evaluation2 = nb.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_nb, "Naive Bayes", "red")

    print("Perceptron")
    perc = percc.Perceptron()
    params_perc = perc.train(dataset.train_X, dataset.train_y)
    print(params_perc.reshape(-1))
    predict = perc.test(dataset.train_X, params_perc)
    evaluation = perc.evaluate(predict, dataset.train_y)
    predict2 = perc.test(dataset.test_X, params_perc)
    evaluation2 = perc.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_perc, "Perceptron", "blue")

    print("MaxEnt LBFGS")
    me = mec_batch.MaxEnt_batch()
    params_me = me.train(dataset.train_X, dataset.train_y)
    print(params_me.reshape(-1))
    predict = me.test(dataset.train_X, params_me)
    evaluation = me.evaluate(predict, dataset.train_y)
    predict2 = me.test(dataset.test_X, params_me)
    evaluation2 = me.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_me, "ME-LBFGS", "green")

    print("MaxEnt Online")
    me_online = mec_online.MaxEnt_online()
    params_me = me_online.train(dataset.train_X, dataset.train_y)
    print(params_me.reshape(-1))
    predict = me_online.test(dataset.train_X, params_me)
    evaluation = me_online.evaluate(predict, dataset.train_y)
    predict2 = me_online.test(dataset.test_X, params_me)
    evaluation2 = me.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_me, "ME-Online", "pink")

    print("MIRA")
    mira = mirac.Mira()
    params_mira = mira.train(dataset.train_X, dataset.train_y)
    print(params_mira.reshape(-1))
    predict = mira.test(dataset.train_X, params_mira)
    evaluation = mira.evaluate(predict, dataset.train_y)
    predict2 = mira.test(dataset.test_X, params_mira)
    evaluation2 = mira.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_mira, "Mira", "orange")

    print("SVM")
    svm = svmc.SVM()
    params_svm = svm.train(dataset.train_X, dataset.train_y)
    print(params_svm.reshape(-1))
    predict = svm.test(dataset.train_X, params_svm)
    evaluation = svm.evaluate(predict, dataset.train_y)
    predict2 = svm.test(dataset.test_X, params_svm)
    evaluation2 = svm.evaluate(predict2, dataset.test_y)
    print("Accuracy train: %f test: %f" % (evaluation, evaluation2))
    fig, axis = dataset.add_line(fig, axis, params_svm, "SVM", "brown")
