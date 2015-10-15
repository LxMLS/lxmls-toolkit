###### Exercises for lab day 4 Parsing.
import sys
sys.path.append('.')

import lxmls.parsing.dependency_parser as depp
import pdb

print "Exercise 4.3.1"

dp = depp.DependencyParser()

dp.read_data("portuguese")

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 4.3.2"

dp.train_perceptron(10)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 4.3.3"

dp.features.use_lexical = True
dp.read_data("portuguese")
dp.train_perceptron(10)
dp.test()

dp.features.use_distance = True
dp.read_data("portuguese")
dp.train_perceptron(10)
dp.test()

dp.features.use_contextual = True
dp.read_data("portuguese")
dp.train_perceptron(10)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 4.3.4"

dp.train_crf_sgd(10, 0.01, 0.1)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 4.3.5"

dp.read_data("english")
dp.train_perceptron(10)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 4.3.6"

dp = depp.DependencyParser()
dp.features.use_lexical = True
dp.features.use_distance = True
dp.features.use_contextual = True
dp.read_data("english")
dp.projective = True
dp.train_perceptron(10)
dp.test()
