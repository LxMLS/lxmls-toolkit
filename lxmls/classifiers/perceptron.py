import sys
import numpy as np
import linear_classifier as lc

class Perceptron(lc.LinearClassifier):

    def __init__(self,nr_epochs = 10,learning_rate = 1, averaged = True):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        # Perceptron Model parameters (values for all training epochs stored)
        self.params_per_epoch = []
	# Training parameters
        self.nr_epochs = nr_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged


    '''
    Trains the parameters of a Perceptron
    '''
    def train(self,x,y):

        # Store original features to be used later
        x_orig = x[:,:]
        # Append a column of ones to the current features
        x = self.add_intercept_term(x)
 
        # Get dimension and number of examples of our input set
        nr_x,nr_f = x.shape
        # Get the number of classes of our output set
        classes = np.unique(y)
        nr_c = classes.shape[0]

        # Initialization of classifier parameters
        w = np.zeros((nr_f,nr_c))
	self.params_per_epoch = [] 
    
        ## Randomize the examples
        perm = np.random.permutation(nr_x)

        # For each training epoch
        for epoch_nr in xrange(self.nr_epochs):
             # For each training example
             for nr in xrange(nr_x):
                #print "iter %i" %( epoch_nr*nr_x + nr)
                # Get one training example index at random
                inst = perm[nr]
		# Predict output using the current model
                y_hat = self.get_label(x[inst:inst+1,:],w)
                # Get true output from training data		
                y_true = y[inst:inst+1,0]
		# If there is a prediction error
                if(y_true != y_hat):
                    # Add feature with respect to the true output
                    w[:,y_true] += self.learning_rate*x[inst:inst+1,:].transpose()
                    # Subtract feature with respect to the predicted output
                    w[:,y_hat] -= self.learning_rate*x[inst:inst+1,:].transpose()

             # Store the parameters for this epoch 
             self.params_per_epoch.append(w.copy())
             # Test accuracy of the model in this epoch of training
             # To keep the test routine happy
             self.trained = True
             # Test
             y_pred = self.test(x_orig,w)
             # Evaluation
             acc = self.evaluate(y,y_pred)
             print "Rounds: %i Accuracy: %f" %( epoch_nr,acc) 
             # We continue training
             self.trained = False

        # Optionaly return the average of all training epoch parameters
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            w=new_w.copy()

        self.trained = True
        
        return w
