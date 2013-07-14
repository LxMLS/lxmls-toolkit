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

class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs = 10, learning_rate = 1.0, averaged = True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        ## Randomize the examples
        num_examples = dataset.size()
        #perm = np.random.permutation(nr_x)
        for epoch in xrange(self.num_epochs):
             num_labels_total = 0
             num_mistakes_total = 0
             for i in xrange(num_examples):
                #print "iter %i" %( round_nr*nr_x + nr)
                #seq = sequence_list[perm[nr]]
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
             self.params_per_epoch.append(self.parameters.copy())   
             acc = 1.0 - float(num_mistakes_total)/float(num_labels_total)
             print "Epoch: %i Accuracy: %f" %(epoch, acc) 
        self.trained = True
        
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            self.parameters = new_w



    def perceptron_update(self, sequence):
        num_labels = 0
        num_mistakes = 0
        
#        y_hat = self.viterbi_decode_log_raw(seq)
        predicted_sequence, _ = self.viterbi_decode(sequence)

        #pdb.set_trace()
#        print predicted_sequence
        y_hat = predicted_sequence.y

        
        # Update initial features.
        y_t_true = sequence.y[0]
        y_t_hat = y_hat[0]

        if y_t_true != y_t_hat:
            true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true)
            self.parameters[true_initial_features] += self.learning_rate
            #print "increasing parameters for pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[seq.y[pos]])
            #print self.feature_class.print_feature_list(truth_node_features)
            hat_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_hat)
            self.parameters[hat_initial_features] -= self.learning_rate
            #print "decreasing parameters pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[y_hat.y[pos][0]])
            #print self.feature_class.print_feature_list(hat_node_features)    

        for pos in xrange(len(sequence.x)):
            y_t_true = sequence.y[pos]
            y_t_hat = y_hat[pos]
            
            # Update emission features.
            num_labels += 1
            if y_t_true != y_t_hat:
                num_mistakes += 1
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                self.parameters[true_emission_features] += self.learning_rate
                #print "increasing parameters for pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[seq.y[pos]])
                #print self.feature_class.print_feature_list(truth_node_features)
                hat_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
                self.parameters[hat_emission_features] -= self.learning_rate
                #print "decreasing parameters pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[y_hat.y[pos][0]])
                #print self.feature_class.print_feature_list(hat_node_features)
                    
                
            if pos > 0:
            ## update bigram features
            ## If true bigram != predicted bigram update bigram features
                prev_y_t_true = sequence.y[pos-1]
#                pdb.set_trace()
                prev_y_t_hat = y_hat[pos-1]
                if(y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat):
                    true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true)
                    self.parameters[true_transition_features] += self.learning_rate                                
                    hat_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_hat, prev_y_t_hat)
                    self.parameters[hat_transition_features] -= self.learning_rate
                    # print "increasing parameters"
                    # print truth_edge_features
                    # print "decreasing parameters"
                    # print hat_edge_features
                        
            #else:
                #print "no errors at pos %i"%(pos)
                
        pos = len(sequence.x)
        y_t_true = sequence.y[pos-1]
        y_t_hat = y_hat[pos-1]

        if y_t_true != y_t_hat:
            true_final_features = self.feature_mapper.get_final_features(sequence, y_t_true)
            self.parameters[true_final_features] += self.learning_rate
            #print "increasing parameters for pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[seq.y[pos]])
            #print self.feature_class.print_feature_list(truth_node_features)
            hat_final_features = self.feature_mapper.get_final_features(sequence, y_t_hat)
            self.parameters[hat_final_features] -= self.learning_rate


        return num_labels, num_mistakes


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
