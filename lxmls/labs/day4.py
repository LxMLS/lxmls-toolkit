###### Exercises for pratica class 1
import parsing.dependency_parser as depp

print "Exercise 6.1"

dp = depp.DependencyParser()

dp.read_data("portuguese")
dp.train_perceptron(10)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 6.2"

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

print "Exercise 6.3"

dp.train_crf_sgd(10, 0.01, 0.1)
dp.test()

goon = raw_input("Enter to go on to next exercise:")

print "Exercise 6.4"

dp.read_data("english")
dp.train_perceptron(10)
dp.test()

