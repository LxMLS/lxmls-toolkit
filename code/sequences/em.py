import sys
import numpy as np
from sequences.viterbi import viterbi
from sequences.forward_backward import forward_backward,sanity_check_forward_backward
from util.my_math_utils import *
from sequences.confusion_matrix import *

class EM():

    def __init__(self,dataset,model):
        self.dataset = dataset
        self.model = model

    def train(self,seq_list,nr_iter=10,smoothing=0,evaluate=True):
        if(evaluate):
            acc = self.evaluate(seq_list)
            print "Init acc %f"%(acc)
            
        for t in xrange(1,nr_iter):
            print "Iter %i"%t
            #E-Step
            total_likelihood = 0
            self.model.clear_counts(smoothing)
            for seq in seq_list.seq_list:
                posteriors,likelihood = self.model.get_posteriors(seq)
                self.model.update_counts(seq,posteriors)
                total_likelihood += likelihood
            #self.model.sanity_check_counts(seq_list,smoothing=smoothing)
            print "Iter: %i Negative Log Likelihood %f"%(t,-1*math.log(total_likelihood))
            #M-Step
            self.model.update_params()
            if(evaluate):
                 ### Evaluate accuracy at this iteration
                acc = self.evaluate(seq_list)
                print "Iter: %i acc %f"%(t,acc)
            

    def evaluate(self,seq_list):
         ### Evaluate accuracy at initial iteration
        pred = self.model.viterbi_decode_corpus(seq_list.seq_list)
        cm = build_confusion_matrix(seq_list.seq_list,pred,len(self.dataset.int_to_pos),self.model.nr_states)
        best = get_best_assignment(cm)
#        print best
        new_pred = []
        for seq in seq_list.seq_list:
            pred_seq = pred[seq.nr]
            new_seq = pred_seq.copy_sequence()
            for i,y_hat in enumerate(new_seq.y):
                y_hat = y_hat[0]
#                print new_seq.y[i]
#                print y_hat
                new_seq.y[i] = best[y_hat]
            new_pred.append(new_seq)
        acc = self.model.evaluate_corpus(seq_list.seq_list,new_pred)
        return acc


