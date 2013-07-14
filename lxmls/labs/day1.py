###### Exercises for labs class 1
import sys, pdb
sys.path.append("." )

import readers.simple_data_set as sds
import readers.sentiment_reader as srs
import classifiers.linear_classifier as lcc
import classifiers.perceptron as percc
import classifiers.mira as mirac
import classifiers.gaussian_naive_bayes as gnbc
import classifiers.multinomial_naive_bayes as mnb
reload(mnb) # this allows you to edit the module and run this script again without rebooting Python
import classifiers.max_ent_batch as mebc
import classifiers.max_ent_online as meoc
import classifiers.svm as svmc
import classifiers.naive_bayes as nb
import run_all_classifiers as run_all_c


#### Exercise 3.1: run all classifiers on 2D data ####

# This instruction generates a simple 2D dataset with two classes.
# Each class is a Gaussian distribution.
# Input parameters (feel free to change them):
    # nr_examples: number of points in the dataset
    # g1: parameters for the first gaussian, of the form:
        # g1 = [[mean_x,mean_y], std]
        # mean_x and mean_y are the x and y coordinates of the mean of the Gaussian
        # std is the standard deviation of the Gaussian
    # g2: parameters for the second gaussian, with a similar form as g1
    # balance: percentage of points in the first gaussian
    # split: fraction of points to use for train, development, and test respectively
sd = sds.SimpleDataSet(nr_examples=100,
                       g1 = [[-1,-1],1],
                       g2 = [[1,1],1],
                       balance=0.5,
                       split=[0.5,0,0.5])

# The above function generates the following variables:
    # sd.train_X
    # sd.train_y
    # sd.test_X
    # sd.test_y


# Plot the data and the Bayes Optimal classifier
fig,axis = sd.plot_data()

# Initialize the Naive Bayes (NB) classifier for Gaussian data
gnb = gnbc.GaussianNaiveBayes()

# Learn the NB parameters from the train data
params_nb_sd = gnb.train(sd.train_X,sd.train_y)

# Use the learned parameters to predict labels for the training data
y_pred_train = gnb.test(sd.train_X, params_nb_sd)

# Compute accuracy on training data from predicted labels and true labels
acc_train = gnb.evaluate(sd.train_y, y_pred_train)

# Use the learned parameters to predict labels for the test data
y_pred_test = gnb.test(sd.test_X, params_nb_sd)

# Compute accuracy on test data from predicted labels and true labels
acc_test = gnb.evaluate(sd.test_y, y_pred_test)

# Add a line to the plot with the line corresponding to the NB classifier
fig,axis = sd.add_line(fig,axis,params_nb_sd,"Naive Bayes","red")

# Print these two accuracies to the terminal
print "Naive Bayes Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the perceptron classifier (instead of Naive Bayes)
perc = percc.Perceptron()
params_perc_sd = perc.train(sd.train_X, sd.train_y)
y_pred_train = perc.test(sd.train_X, params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X,params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
fig,axis = sd.add_line(fig,axis,params_perc_sd,"Perceptron","blue")
print "Perceptron Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the MIRA classifier
mira = mirac.Mira()
params_mira_sd = mira.train(sd.train_X,sd.train_y)
y_pred_train = mira.test(sd.train_X,params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X,params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
fig,axis = sd.add_line(fig,axis,params_mira_sd,"Mira","green")
print "Mira Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the Maximum Entropy classifier, batch version
me_lbfgs = mebc.MaxEnt_batch()
params_meb_sd = me_lbfgs.train(sd.train_X,sd.train_y)
y_pred_train = me_lbfgs.test(sd.train_X,params_meb_sd)
acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(sd.test_X,params_meb_sd)
acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
fig,axis = sd.add_line(fig,axis,params_meb_sd,"Max-Ent-Batch","orange")
print "Max-Ent batch Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the Maximum Entropy classifier, online version
me_sgd = meoc.MaxEnt_online()
params_meo_sd = me_sgd.train(sd.train_X,sd.train_y)
y_pred_train = me_sgd.test(sd.train_X,params_meo_sd)
acc_train = me_sgd.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_sgd.test(sd.test_X,params_meo_sd)
acc_test = me_sgd.evaluate(sd.test_y, y_pred_test)
fig,axis = sd.add_line(fig,axis,params_meo_sd,"Max-Ent-Online","magenta")
print "Max-Ent Online Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the SVM classifier
svm = svmc.SVM()
params_svm_sd = svm.train(sd.train_X,sd.train_y)
y_pred_train = svm.test(sd.train_X,params_svm_sd)
acc_train = svm.evaluate(sd.train_y, y_pred_train)
y_pred_test = svm.test(sd.test_X,params_svm_sd)
acc_test = svm.evaluate(sd.test_y, y_pred_test)
fig,axis = sd.add_line(fig,axis,params_svm_sd,"SVM","yellow")
print "SVM Online Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
print

####### End of exercise 3.1 #########


####### Exercise 3.2: implement Naive Bayes for multinomial data ########

# Read the book review data
scr = srs.SentimentCorpus("books")

# Initialize the Naive Bayes classifier for multinomial data
mnb = mnb.MultinomialNaiveBayes()

# Learn the NB parameters from the train data
params_nb_sc = mnb.train(scr.train_X,scr.train_y)

# Use the learned parameters to predict labels for the training data
y_pred_train = mnb.test(scr.train_X,params_nb_sc)

# Compute accuracy on training data from predicted labels and true labels
acc_train = mnb.evaluate(scr.train_y, y_pred_train)

# Use the learned parameters to predict labels for the test data
y_pred_test = mnb.test(scr.test_X,params_nb_sc)

# Compute accuracy on test data from predicted labels and true labels
acc_test = mnb.evaluate(scr.test_y, y_pred_test)

# Print these two accuracies to the terminal
# You should get 0.656250 on the train set and 0.622500 on the test set
print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the perceptron classifier (instead of Naive Bayes)
params_perc_sc = perc.train(scr.train_X,scr.train_y)
y_pred_train = perc.test(scr.train_X,params_perc_sc)
acc_train = perc.evaluate(scr.train_y, y_pred_train)
y_pred_test = perc.test(scr.test_X,params_perc_sc)
acc_test = perc.evaluate(scr.test_y, y_pred_test)
print "Perceptron Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the MIRA classifier
params_mira_sc = mira.train(scr.train_X,scr.train_y)
y_pred_train = mira.test(scr.train_X,params_mira_sc)
acc_train = mira.evaluate(scr.train_y, y_pred_train)
y_pred_test = mira.test(scr.test_X,params_mira_sc)
acc_test = mira.evaluate(scr.test_y, y_pred_test)
print "Mira Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the Maximum Entropy classifier, batch version
params_meb_sc = me_lbfgs.train(scr.train_X,scr.train_y)
y_pred_train = me_lbfgs.test(scr.train_X,params_meb_sc)
acc_train = me_lbfgs.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(scr.test_X,params_meb_sc)
acc_test = me_lbfgs.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Batch Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the Maximum Entropy classifier, online version
params_meo_sc = me_sgd.train(scr.train_X,scr.train_y)
y_pred_train = me_sgd.test(scr.train_X,params_meo_sc)
acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_sgd.test(scr.test_X,params_meo_sc)
acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print

# Same as above, but for the SVM classifier
params_svm_sc = svm.train(scr.train_X,scr.train_y)
y_pred_train = svm.test(scr.train_X,params_svm_sc)
acc_train = svm.evaluate(scr.train_y, y_pred_train)
y_pred_test = svm.test(scr.test_X,params_svm_sc)
acc_test = svm.evaluate(scr.test_y, y_pred_test)
print "SVM Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
print
