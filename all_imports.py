'''
Test if all import work. Run this from the lxmls-toolkit folder
'''
import sys 
import numpy
import matplotlib.pyplot
import math
import lxmls.sequences.structured_perceptron
import lxmls.sequences.id_feature
import lxmls.sequences.hmm
import lxmls.sequences.extended_feature
import lxmls.sequences.crf_online
import lxmls.sequences.confusion_matrix
import lxmls.run_all_classifiers
import lxmls.readers.simple_sequence
import lxmls.readers.simple_data_set
import lxmls.readers.sentiment_reader
import lxmls.readers.pos_corpus
import lxmls.parsing.dependency_parser
import lxmls.classifiers.svm
import lxmls.classifiers.perceptron
import lxmls.classifiers.naive_bayes
import lxmls.classifiers.naive_bayes
import lxmls.classifiers.multinomial_naive_bayes
import lxmls.classifiers.mira
import lxmls.classifiers.max_ent_online
import lxmls.classifiers.max_ent_online
import lxmls.classifiers.max_ent_batch
import lxmls.classifiers.max_ent_batch
import lxmls.classifiers.linear_classifier
import lxmls.classifiers.gaussian_naive_bayes
import lxmls.sequences.log_domain 
