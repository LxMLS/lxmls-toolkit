###### Exercises for pratica class 1

from lxmls.readers import simple_data_set as sds
from lxmls.readers import sentiment_reader as srs
from lxmls.classifiers import perceptron as percc
from lxmls.classifiers import mira as mirac
from lxmls.classifiers import gaussian_naive_bayes as gnbc
from lxmls.classifiers import multinomial_naive_bayes as mnb
from lxmls.classifiers import max_ent_batch as mebc
from lxmls.classifiers import max_ent_online as meoc
from lxmls.classifiers import svm as svmc


#### Exercise 3.1 ####
print "Exercise 3.1"
sd = sds.SimpleDataSet(nr_examples=100,g1 = [[-1,-1],1], g2 = [[1,1],1],balance=0.5,split=[0.5,0,0.5])
gnb = gnbc.GaussianNaiveBayes()
params_nb_sd = gnb.train(sd.train_X,sd.train_y)
print "Estimated Means"
print gnb.means
print "Estimated Priors"
print gnb.prior
y_pred_train = gnb.test(sd.train_X,params_nb_sd)
acc_train = gnb.evaluate(sd.train_y, y_pred_train)
y_pred_test = gnb.test(sd.test_X,params_nb_sd)
acc_test = gnb.evaluate(sd.test_y, y_pred_test)
print "Gaussian Naive Bayes Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.2"
scr = srs.SentimentCorpus("books")
mnb = mnb.MultinomialNaiveBayes()
params_nb_sc = mnb.train(scr.train_X,scr.train_y)
y_pred_train = mnb.test(scr.train_X,params_nb_sc)
acc_train = mnb.evaluate(scr.train_y, y_pred_train)
y_pred_test = mnb.test(scr.test_X,params_nb_sc)
acc_test = mnb.evaluate(scr.test_y, y_pred_test)
print "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")


print "Exercise 3.3"
fig,axis = sd.plot_data()
fig,axis = sd.add_line(fig,axis,params_nb_sd,"Naive Bayes","red")
#goon = raw_input("Enter to go on to next exercise:")
#### Exercices not in guide ####


print "Exercise 3.4 1"
perc = percc.Perceptron()
params_perc_sd = perc.train(sd.train_X,sd.train_y)
y_pred_train = perc.test(sd.train_X,params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X,params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
print "Perceptron Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.4 2"
fig,axis = sd.add_line(fig,axis,params_perc_sd,"Perceptron","blue")
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.4 3"
params_perc_sc = perc.train(scr.train_X,scr.train_y)
y_pred_train = perc.test(scr.train_X,params_perc_sc)
acc_train = perc.evaluate(scr.train_y, y_pred_train)
y_pred_test = perc.test(scr.test_X,params_perc_sc)
acc_test = perc.evaluate(scr.test_y, y_pred_test)
print "Perceptron Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")


print "Exercise 3.5 1"
mira = mirac.Mira()
params_mira_sd = mira.train(sd.train_X,sd.train_y)
y_pred_train = mira.test(sd.train_X,params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X,params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
print "Mira Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.5 2"
fig,axis = sd.add_line(fig,axis,params_mira_sd,"Mira","green")
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.5 3"
params_mira_sc = mira.train(scr.train_X,scr.train_y)
y_pred_train = mira.test(scr.train_X,params_mira_sc)
acc_train = mira.evaluate(scr.train_y, y_pred_train)
y_pred_test = mira.test(scr.test_X,params_mira_sc)
acc_test = mira.evaluate(scr.test_y, y_pred_test)
print "Mira Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")



print "Exercise 3.6 1"
me_lbfgs = mebc.MaxEnt_batch()
params_meb_sd = me_lbfgs.train(sd.train_X,sd.train_y)
y_pred_train = me_lbfgs.test(sd.train_X,params_meb_sd)
acc_train = me_lbfgs.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(sd.test_X,params_meb_sd)
acc_test = me_lbfgs.evaluate(sd.test_y, y_pred_test)
print "Max-Ent batch Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.6 2"
fig,axis = sd.add_line(fig,axis,params_meb_sd,"Max-Ent-Batch","orange")
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.6 3"
params_meb_sc = me_lbfgs.train(scr.train_X,scr.train_y)
y_pred_train = me_lbfgs.test(scr.train_X,params_meb_sc)
acc_train = me_lbfgs.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_lbfgs.test(scr.test_X,params_meb_sc)
acc_test = me_lbfgs.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Batch Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")


print "Exercise 3.7 1"
me_sgd = meoc.MaxEnt_online()
params_meo_sd = me_sgd.train(sd.train_X,sd.train_y)
y_pred_train = me_sgd.test(sd.train_X,params_meo_sd)
acc_train = me_sgd.evaluate(sd.train_y, y_pred_train)
y_pred_test = me_sgd.test(sd.test_X,params_meo_sd)
acc_test = me_sgd.evaluate(sd.test_y, y_pred_test)
print "Max-Ent Online Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.7 2"
fig,axis = sd.add_line(fig,axis,params_meo_sd,"Max-Ent-Batch","orange")
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.7 3"
params_meo_sc = me_sgd.train(scr.train_X,scr.train_y)
y_pred_train = me_sgd.test(scr.train_X,params_meo_sc)
acc_train = me_sgd.evaluate(scr.train_y, y_pred_train)
y_pred_test = me_sgd.test(scr.test_X,params_meo_sc)
acc_test = me_sgd.evaluate(scr.test_y, y_pred_test)
print "Max-Ent Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")


print "Exercise 3.8 1"
svm = svmc.SVM()
params_svm_sd = svm.train(sd.train_X,sd.train_y)
y_pred_train = svm.test(sd.train_X,params_svm_sd)
acc_train = svm.evaluate(sd.train_y, y_pred_train)
y_pred_test = svm.test(sd.test_X,params_svm_sd)
acc_test = svm.evaluate(sd.test_y, y_pred_test)
print "SVM Online Simple Dataset Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.8 2"
fig,axis = sd.add_line(fig,axis,params_svm_sd,"Max-Ent-Batch","orange")
#goon = raw_input("Enter to go on to next exercise:")

print "Exercise 3.8 3"
params_svm_sc = svm.train(scr.train_X,scr.train_y)
y_pred_train = svm.test(scr.train_X,params_svm_sc)
acc_train = svm.evaluate(scr.train_y, y_pred_train)
y_pred_test = svm.test(scr.test_X,params_svm_sc)
acc_test = svm.evaluate(scr.test_y, y_pred_test)
print "SVM Online Amazon Sentiment Accuracy train: %f test: %f"%(acc_train,acc_test)
#goon = raw_input("Enter to go on to next exercise:")


####
## Exercise 1.1
# print "Exercise 1.1" 
# sd = sds.SimpleDataSet(g1 = [[-1,-1],0.5], g2 = [[1,1],0.5])
# run_all_c.run_all_classifiers(sd)



# ####


# ####
# ## Exercise 1.1
# print "Exercise 1.1" 
# sd = sds.SimpleDataSet(g1 = [[-1,-1],1], g2 = [[1,1],1])
# run_all_c.run_all_classifiers(sd)


# ####

# ## Exercise 1.3
# print "Exercise 1.3" 
# sd = sds.SimpleDataSet(g1 = [[-1,-1],1], g2 = [[1,1],1],balance=0.1)
# run_all_c.run_all_classifiers(sd)

