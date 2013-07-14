#import sys
#import numpy as np
#import matplotlib.pyplot as plt
#import scipy.optimize.lbfgsb as opt2
#sys.path.append("util/" )

import sys
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append("util/" )
#from my_math_utils import *
##from forward_backward import forward_backward,sanity_check_forward_backward
#sys.path.append("sequences/" )
import discriminative_sequence_classifier as dsc

import pdb

class CRFOnline(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self, observation_labels, state_labels, feature_mapper, 
                 regularizer=0.00001,
                 num_epochs=10, initial_learning_rate=10.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.regularizer = regularizer
        self.num_epochs = num_epochs
        self.initial_learning_rate = initial_learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        ## Randomize the examples
        num_examples = dataset.size()
        t = 0
        #perm = np.random.permutation(nr_x)
        for epoch in xrange(self.num_epochs):
             objective_value = 0.0
             for i in xrange(num_examples):
                eta = self.initial_learning_rate / np.sqrt(float(t+1))
#                eta = self.initial_learning_rate / float(t+1)
                #print "iter %i" %( round_nr*nr_x + nr)
                #seq = sequence_list[perm[nr]]
                sequence = dataset.seq_list[i]
                objective_value += self.gradient_update(sequence, eta)
                t += 1
             self.params_per_epoch.append(self.parameters.copy())   
             objective_value /= num_examples
             print "Epoch: %i Objective value: %f" %(epoch, objective_value) 
        self.trained = True
        
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            self.parameters = new_w



    def gradient_update(self, sequence, eta):
        objective_value = 0.0
        num_states = self.get_num_states() # Number of states.
        
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = \
            self.compute_scores(sequence)

        state_posteriors, transition_posteriors, log_likelihood = \
            self.compute_posteriors(initial_scores, transition_scores,
                                    final_scores, emission_scores)

        # Add the score of the true sequence.            
        objective_value += self.compute_output_score(sequence.y,
                                                     initial_scores,
                                                     transition_scores,
                                                     final_scores,
                                                     emission_scores)

        # Subtract the log-partition function.
        # Convince yourself that this is indeed the value of log Z.            
        objective_value -= log_likelihood
        
        # Add the squared norm of the parameter vector.            
        objective_value -= 0.5 * self.regularizer * np.dot(self.parameters, self.parameters)

        # Scale the parameter vector.
        self.parameters *= (1.0 - self.regularizer * eta)
        
        # Update initial features.
        y_t_true = sequence.y[0]
        true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true)
        self.parameters[true_initial_features] += eta
        for state in xrange(num_states):
            state_initial_features = self.feature_mapper.get_initial_features(sequence, state)
            self.parameters[state_initial_features] -= eta * state_posteriors[0, state]
                    
        for pos in xrange(len(sequence.x)):
            # Update emission features.
            y_t_true = sequence.y[pos]
            true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
            self.parameters[true_emission_features] += eta
            for state in xrange(num_states):
                state_emission_features = self.feature_mapper.get_emission_features(sequence, pos, state)
                self.parameters[state_emission_features] -= eta * state_posteriors[pos, state]
                                    
            if pos > 0:
                # Update transition features.
                prev_y_t_true = sequence.y[pos-1]
                true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true)
                self.parameters[true_transition_features] += eta                                
                for state in xrange(num_states):
                    for prev_state in xrange(num_states):
                        state_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, state, prev_state)
                        self.parameters[state_transition_features] -= \
                            eta * transition_posteriors[pos-1, state, prev_state]
                
        pos = len(sequence.x)
        y_t_true = sequence.y[pos-1]
        true_final_features = self.feature_mapper.get_final_features(sequence, y_t_true)
        self.parameters[true_final_features] += eta
        for state in xrange(num_states):
            state_final_features = self.feature_mapper.get_final_features(sequence, state)
            self.parameters[state_final_features] -= eta * state_posteriors[pos-1, state]

        return objective_value


    def save_model(self,dir):
        fn = open(dir+"parameters.txt",'w')
        for p_id,p in enumerate(self.parameters):
            fn.write("%i\t%f\n"%(p_id,p))
        fn.close()
    
    def load_model(self,dir):
        fn = open(dir+"parameters.txt",'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
