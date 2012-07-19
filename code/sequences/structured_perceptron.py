import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize.lbfgsb as opt2
sys.path.append("util/" )


import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("util/" )
from my_math_utils import *
from forward_backward import forward_backward,sanity_check_forward_backward
sys.path.append("sequences/" )
import discriminative_sequence_classifier as dsc

class StructuredPercetron(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self,dataset,feature_class,nr_rounds = 10,learning_rate = 1, averaged = True):
        dsc.DiscriminativeSequenceClassifier.__init__(self,dataset,feature_class)
        self.nr_rounds = nr_rounds
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_round = []

    def train_supervised(self,sequence_list):
        self.parameters = np.zeros(self.feature_class.nr_feats,dtype=float)
        ## Randomize the examples
        nr_x = len(sequence_list)
        #perm = np.random.permutation(nr_x)
        for round_nr in xrange(self.nr_rounds):
             incorrect = 0
             total = 0
             for nr in xrange(nr_x):
                #print "iter %i" %( round_nr*nr_x + nr)
                #seq = sequence_list[perm[nr]]
                seq = sequence_list[nr]
                total,incorrect = self.process_one_example(seq,total,incorrect)
             self.params_per_round.append(self.parameters.copy())   
             acc = 1.0 - 1.0*incorrect/total
             print "Epoch: %i Accuracy: %f" %( round_nr,acc) 
        self.trained = True
        
        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_round:
                new_w += old_w
            new_w = new_w / len(self.params_per_round)
            self.parameters = new_w



    def process_one_example(self,seq,total,incorrect):
        
        y_hat = self.viterbi_decode_log_raw(seq)
        ## Update features
        for pos in xrange(len(seq.x)):
            y_t_true = seq.y[pos]
            y_t_hat = y_hat[pos][0]
            total += 1
            if(y_t_true != y_t_hat):
                incorrect += 1
                truth_node_features = self.feature_class.get_node_features(seq,pos,y_t_true)
                self.parameters[truth_node_features] += self.learning_rate
                #print "increasing parameters for pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[seq.y[pos]])
                #print self.feature_class.print_feature_list(truth_node_features)
                hat_node_features = self.feature_class.get_node_features(seq,pos,y_t_hat)
                self.parameters[hat_node_features] -= self.learning_rate
                #print "decreasing parameters pos %i word %s tag %s"%(pos, self.dataset.int_to_word[seq.x[pos]],self.dataset.int_to_pos[y_hat.y[pos][0]])
                #print self.feature_class.print_feature_list(hat_node_features)
            if(pos > 0):
            ## update bigram features
            ## If true bigram != predicted bigram update bigram features
                prev_y_t_true = seq.y[pos-1]
                prev_y_t_hat = y_hat[pos-1][0]
                if(y_t_true != y_t_hat or  prev_y_t_true !=  prev_y_t_hat):
                    truth_edge_features = self.feature_class.get_edge_features(seq,pos,y_t_true,prev_y_t_true)
                    self.parameters[truth_edge_features] += self.learning_rate                                
                    hat_edge_features = self.feature_class.get_edge_features(seq,pos,y_t_hat,prev_y_t_hat)
                    self.parameters[hat_edge_features] -= self.learning_rate
                    # print "increasing parameters"
                    # print truth_edge_features
                    # print "decreasing parameters"
                    # print hat_edge_features
                        
            #else:
                #print "no errors at pos %i"%(pos)
        return total,incorrect


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
