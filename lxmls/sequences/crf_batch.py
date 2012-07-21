import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



from util.my_math_utils import *
from viterbi import viterbi
from forward_backward import forward_backward,sanity_check_forward_backward
import discriminative_sequence_classifier as dsc

class CRF_batch(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self,dataset,feature_class,regularizer=0.01):
        dsc.DiscriminativeSequenceClassifier.__init__(self,dataset,feature_class)
        self.regularizer = regularizer

    def train_supervised(self,sequence_list):
        self.parameters = np.zeros(self.feature_class.nr_feats)
        emp_counts = self.get_empirical_counts(sequence_list)
        params,_,d = optimize.fmin_l_bfgs_b(self.get_objective,self.parameters,args=[sequence_list,emp_counts],factr = 1e14,maxfun = 500,iprint = 1,pgtol=1e-5)
        self.parameters = params
        self.trained = True
        return params



    def get_objective(self,parameters,sequence_list,emp_counts):
        self.parameters = parameters
        gradient = np.zeros(parameters.shape)
        gradient += emp_counts
        objective = 0
        likelihoods = 0
        exp_counts = np.zeros(parameters.shape)            
        for sequence in sequence_list:
            seq_obj,seq_lik = self.get_objective_seq(parameters,sequence,exp_counts)
            objective += seq_obj
            likelihoods += seq_lik
        objective -= 0.5*self.regularizer*np.dot(parameters,parameters)
        if(likelihoods != 0):
            objective -= np.log(likelihoods)
        else:
            print "likelihoods == 0"
        gradient -= self.regularizer*parameters
        gradient -= exp_counts

        ##Since we are minizing we need to multiply both the objective and gradient by -1
        objective = -1*objective
        gradient = gradient*-1
#        print "Objective: %f"%objective
        #print gradient
 #       print "Gradient norm: %f"%np.sqrt(np.dot(gradient,gradient))
        ## Sicne we are minimizing and not maximizing
 
        return objective,gradient



    def test_get_objective_seq(self,parameters,seq,times):
        exp_counts = np.zeros(parameters.shape)
        for i in xrange(times):
            self.get_objective_seq(parameters,seq,exp_counts)


    
    def test(self):
        a = [1,2,3]
        b = np.arange(1,2000,1)
        c = 0
        for i in xrange(1000000):
            c += b[a]


    def test2(self):
        a = [1,2,3]
        b = np.arange(1,2000,1)
        c = 0
        for i in xrange(1000000):
            for j in a:
                c += b[j]


    def get_objective_seq(self,parameters,seq,exp_counts):
         #print seq.nr
         nr_states = self.nr_states
         node_potentials,edge_potentials = self.build_potentials(seq)
         forward,backward = forward_backward(node_potentials,edge_potentials)
         H,N = forward.shape
         likelihood = np.sum(forward[:,N-1])
         #node_posteriors = self.get_node_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
         #edge_posteriors = self.get_edge_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
         seq_objective = 0
         for pos in xrange(N): 
             true_y = seq.y[pos]
             for state in xrange(H):
                 node_f_list = self.feature_class.get_node_features(seq,pos,state)
                 backward_aux = backward[state,pos]
                 forward_aux = forward[state,pos]
                 forward_aux_div_likelihood = forward_aux/likelihood

                 ##Iterate over feature indexes
                 prob_aux = forward_aux_div_likelihood*backward_aux
                 for fi in node_f_list:
                     ## For the objective add both the node features and edge feature dot the parameters for the true observation
                     if(state == true_y):
                         seq_objective += parameters[fi]
                     ## For the gradient add the node_posterior ##Compute node posteriors on the fly
                     exp_counts[fi] += prob_aux
                 #Handle transitions
                 if(pos < N-1):
                     true_next_y = seq.y[pos+1]
                     for next_state in xrange(H):
                         backward_aux2 = backward[next_state,pos+1]
                         node_pot_aux = node_potentials[next_state,pos+1]
                         edge_f_list = self.feature_class.get_edge_features(seq,pos+1,next_state,state)
                         ## For the gradient add the edge_posterior
                         edge_aux = edge_potentials[state,next_state,pos]
                         prob_aux = forward_aux_div_likelihood*edge_aux*node_pot_aux*backward_aux2
                         for fi in edge_f_list: 
                             ## For the objective add both the node features and edge feature dot the parameters for the true observation
                             if(next_state == true_next_y):
                                 seq_objective += parameters[fi]
                             exp_counts[fi] += prob_aux
         return seq_objective,likelihood





    def get_empirical_counts(self,sequence_list):
        emp_counts = np.zeros(self.feature_class.nr_feats)
        for seq_node_features,seq_edge_features in self.feature_class.feature_list:
            for f_l in seq_node_features:
                for f in f_l:
                    emp_counts[f] += 1
            for f_l in seq_edge_features:
                for f in f_l:
                    emp_counts[f] += 1
        return emp_counts


    def print_node_posteriors(self,seq,node_posteriors):
        print seq.nr
        print seq
        H,N = node_posteriors.shape
        txt = []
        for i in xrange(H):
            txt.append("%s\t"%self.dataset.int_to_pos[i])
        
        for pos in xrange(N):
            for i in xrange(H):
                txt[i] += "%f\t"%node_posteriors[i,pos]
        for i in xrange(H):
            print txt[i]
        print ""
        print ""

    def posterior_decode(self,seq):
        posteriors = self.get_node_posteriors(seq)
        self.print_node_posteriors(seq,posteriors)
        res =  np.argmax(posteriors,axis=0)
        new_seq =  seq.copy_sequence()
        new_seq.y = res
        return new_seq
