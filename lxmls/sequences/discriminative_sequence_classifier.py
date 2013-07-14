
#import sys
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append("util/" )

#from my_math_utils import *
#from viterbi import run_viterbi
#from viterbi_2 import viterbi_log
#from forward_backward import forward_backward,sanity_check_forward_backward

import sequence_classifier as sc

import pdb

class DiscriminativeSequenceClassifier(sc.SequenceClassifier):

    def __init__(self, observation_labels, state_labels, feature_mapper):
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)
        
        # Set feature mapper and initialize parameters.
        self.feature_mapper = feature_mapper
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        
        
    ################################
    ##  Build the node and edge potentials
    ## node - f(t,y_t,X)*w
    ## edge - f(t,y_t,y_(t-1),X)*w
    ## Only supports binary features representation
    ## If we have an HMM with 4 positions and transitins
    ## a - b - c - d
    ## the edge potentials have at position:
    ## 0 a - b
    ## 1 b - c
    ################################
    def compute_scores(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros(num_states)
        transition_scores = np.zeros([length-1, num_states, num_states])
        final_scores = np.zeros(num_states)

        # Initial position.
        for tag_id in xrange(num_states):
             initial_features = self.feature_mapper.get_initial_features(sequence, tag_id)
             score = 0.0
             for feat_id in initial_features:
                 score += self.parameters[feat_id]
             initial_scores[tag_id] = score
        
        # Intermediate position.
        for pos in xrange(length):
            for tag_id in xrange(num_states):
                 emission_features = self.feature_mapper.get_emission_features(sequence, pos, tag_id)
                 score = 0.0
                 for feat_id in emission_features:
                     score += self.parameters[feat_id]
                 emission_scores[pos, tag_id] = score
            if pos > 0: 
                for tag_id in xrange(num_states):
                    for prev_tag_id in xrange(num_states):
                        transition_features = self.feature_mapper.get_transition_features(sequence, pos, tag_id, prev_tag_id)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[pos-1, tag_id, prev_tag_id] = score

        # Final position.
        for prev_tag_id in xrange(num_states):
             final_features = self.feature_mapper.get_final_features(sequence, prev_tag_id)
             score = 0.0
             for feat_id in final_features:
                 score += self.parameters[feat_id]
             final_scores[prev_tag_id] = score

        return initial_scores, transition_scores, final_scores, emission_scores

#    def build_potentials(self,sequence):
#        nr_states = self.nr_states
#        nr_pos = len(sequence.x)
#        #Node potentials are indexed by state and position
#        node_potentials = np.ones([nr_states,nr_pos],dtype=float)
#        #Edge potentials are indexed by tag,next_tag, pos where pos is the position of tag
#        edge_potentials = np.ones([nr_states,nr_states,nr_pos-1],dtype=float)
#        ## We will assume that transition features do not depend on any X information so
#        ## they are the same for all positions this will speed up the code but if the features
#        ## change need to uncomment the code in the main loop and comment here
#        for tag_id in xrange(nr_states):
#            for prev_tag_id in xrange(nr_states):
#                edge_f_list = self.feature_class.get_edge_features(sequence,1,tag_id,prev_tag_id)
#                #print "Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
#                #print edge_f_list
#                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
#                dot_w_f_edge = 0
#                for fi in edge_f_list:
#                    dot_w_f_edge += self.parameters[fi]
#                edge_potentials[prev_tag_id,tag_id,:] = self.my_exp(dot_w_f_edge)
#        #print "edge_potentials"
#        #print edge_potentials
#        ## Add first position
#        for tag_id in xrange(nr_states):
#            edge_f_list = self.feature_class.get_edge_features(sequence,0,tag_id,-1)
#            dot_w_f_edge = 0
#            for fi in edge_f_list:
#                dot_w_f_edge += self.parameters[fi]
#            node_potentials[tag_id,0] *= self.my_exp(dot_w_f_edge)
#        #Add last position
#        last_pos = len(sequence.x)
#        for tag_id in xrange(nr_states):
#                edge_f_list = self.feature_class.get_edge_features(sequence,last_pos,-1,tag_id)
#                dot_w_f_edge = 0
#                for fi in edge_f_list:
#                    dot_w_f_edge += self.parameters[fi]
#                node_potentials[tag_id,last_pos -1] *= self.my_exp(dot_w_f_edge)
#        #print "edge_potentials after final"
#        #print edge_potentials
#        for pos,word_id in enumerate(sequence.x):
#            for tag_id in xrange(nr_states):
#                #f(t,y_t,X)
#                node_f_list = self.feature_class.get_node_features(sequence,pos,tag_id)
#                #print "Node list: pos:%i tag:%i"%(pos,tag_id)
#                #print node_f_list
#                ##w*f - since f is only 0/1 its just the sum of active features
#                
#                #dot_w_f_node = np.sum(self.parameters[node_f_list,:])
#                dot_w_f_node = 0
#                for fi in node_f_list:
#                    dot_w_f_node += self.parameters[fi]
#                node_potentials[tag_id,pos] *= self.my_exp(dot_w_f_node)
#                
#                ##Note this code is commented since we are assuming that transition features do not depende on X information
#                # if(pos > 0):
#                #     for prev_tag_id in all_tags:
#                #         edge_f_list = self.feature_class.get_edge_features(sequence,pos,tag_id,prev_tag_id)
#                #         dot_w_f_edge = np.sum(self.parameters[edge_f_list,:])
#                #         edge_potentials[prev_tag_id,tag_id,pos-1] = np.exp(dot_w_f_edge)
#        return node_potentials,edge_potentials


    ################################
    ##  Build the node and edge potentials on log space.
    ## node - f(t,y_t,X)*w
    ## edge - f(t,y_t,y_(t-1),X)*w
    ## Only supports binary features representation
    ## If we have an HMM with 4 positions and transitins
    ## a - b - c - d
    ## the edge potentials have at position:
    ## 0 a - b
    ## 1 b - c
    ################################
#    def build_potentials_log(self,sequence):
#        nr_states = self.nr_states
#        nr_pos = len(sequence.x)
#        node_potentials = np.zeros([nr_states,nr_pos],dtype=float)
#        edge_potentials = np.zeros([nr_states,nr_states,nr_pos-1],dtype=float)
#
#
#        ## We will assume that transition features do not depend on any X information so
#        ## they are the same for all positions this will speed up the code but if the features
#        ## change need to uncomment the code in the main loop and comment here
#        for tag_id in xrange(nr_states):
#            for prev_tag_id in xrange(nr_states):
#                edge_f_list = self.feature_class.get_edge_features(sequence,1,tag_id,prev_tag_id)
#                #print "Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
#                #print edge_f_list
#                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
#                dot_w_f_edge = 0
#                for fi in edge_f_list:
#                    dot_w_f_edge += self.parameters[fi]
#                edge_potentials[prev_tag_id,tag_id,:-1] = dot_w_f_edge
#        #print "edge_potentials"
#        #print edge_potentials
#        ## Add last position
#        last_pos = len(sequence.x)-1
#        for tag_id in xrange(nr_states):
#            for prev_tag_id in xrange(nr_states):
#                #print "Final Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
#                #print edge_f_list
#                edge_f_list = self.feature_class.get_edge_features(sequence,last_pos,tag_id,prev_tag_id)
#                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
#                dot_w_f_edge = 0
#                for fi in edge_f_list:
#                    dot_w_f_edge += self.parameters[fi]
#                edge_potentials[prev_tag_id,tag_id,last_pos -1] = dot_w_f_edge
#                
#                    
#        #print "edge_potentials after final"
#        #print edge_potentials
#        for pos,word_id in enumerate(sequence.x):
#            for tag_id in xrange(nr_states):
#                #f(t,y_t,X)
#                node_f_list = self.feature_class.get_node_features(sequence,pos,tag_id)
#                #print "Node list: pos:%i tag:%i"%(pos,tag_id)
#                #print node_f_list
#                ##w*f - since f is only 0/1 its just the sum of active features
#                
#                #dot_w_f_node = np.sum(self.parameters[node_f_list,:])
#                dot_w_f_node = 0
#                for fi in node_f_list:
#                    dot_w_f_node += self.parameters[fi]
#                node_potentials[tag_id,pos] = dot_w_f_node
#                ##Note this code is commented since we are assuming that transition features do not depende on X information
#                # if(pos > 0):
#                #     for prev_tag_id in all_tags:
#                #         edge_f_list = self.feature_class.get_edge_features(sequence,pos,tag_id,prev_tag_id)
#                #         dot_w_f_edge = np.sum(self.parameters[edge_f_list,:])
#                #         edge_potentials[prev_tag_id,tag_id,pos-1] = np.exp(dot_w_f_edge)
#        return node_potentials,edge_potentials

    
#    def forward_backward(self,seq):
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        forward,backward = forward_backward(node_potentials,edge_potentials)
#        #print sanity_check_forward_backward(forward,backward)
#        return forward,backward
        

#    ###############
#    ## Returns the node posterios
#    ####################
#    def get_node_posteriors(self,seq):
#        nr_states = self.nr_states
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        forward,backward = forward_backward(node_potentials,edge_potentials)
#        #print sanity_check_forward_backward(forward,backward)
#        H,N = forward.shape
#        likelihood = np.sum(forward[:,N-1])
#        #print likelihood
#        return self.get_node_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
#        
#
#    def get_node_posteriors_aux(self,seq,forward,backward,node_potentials,edge_potentials,likelihood):
#        H,N = forward.shape
#        posteriors = np.zeros([H,N],dtype=float)
#        
#        for pos in  xrange(N):
#            for current_state in xrange(H):
#                posteriors[current_state,pos] = forward[current_state,pos]*backward[current_state,pos]/likelihood
#        return posteriors
#
#    def get_edge_posteriors(self,seq):
#        nr_states = self.nr_states
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        forward,backward = forward_backward(node_potentials,edge_potentials)
#        H,N = forward.shape
#        likelihood = np.sum(forward[:,N-1])
#        return self.get_edge_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
#        
#    def get_edge_posteriors_aux(self,seq,forward,backward,node_potentials,edge_potentials,likelihood):
#        H,N = forward.shape
#        edge_posteriors = np.zeros([H,H,N-1],dtype=float)
#        for pos in xrange(N-1):
#            for prev_state in xrange(H):
#                for state in xrange(H):
#                    edge_posteriors[prev_state,state,pos] = forward[prev_state,pos]*edge_potentials[prev_state,state,pos]*node_potentials[state,pos+1]*backward[state,pos+1]/likelihood 
#        return edge_posteriors
#
#    def posterior_decode(self,seq):
#        posteriors = self.get_node_posteriors(seq)
#        print posteriors
#        res =  np.argmax(posteriors,axis=0)
#        new_seq =  seq.copy_sequence()
#        new_seq.y = res
#        return new_seq
#    
#    def posterior_decode_corpus(self,seq_list):
#        predictions = []
#        for seq in seq_list:
#            predictions.append(self.posterior_decode(seq))
#        return predictions
#
#
#    
#    
#    def viterbi_decode(self,seq):
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        viterbi_path,_ = viterbi(node_potentials,edge_potentials)
#        res =  viterbi_path
#        new_seq =  seq.update_from_sequence(res)
#        return new_seq
#
#
#
#    def viterbi_decode_corpus(self,seq_list):
#        predictions = []
#        for seq in seq_list:
#            predictions.append(self.viterbi_decode(seq))
#        return predictions
#
#
#    def viterbi_decode_log(self,seq):
#        node_potentials,edge_potentials = self.build_potentials_log(seq)
#        viterbi_path,_ = viterbi_log(node_potentials,edge_potentials)
#        res =  viterbi_path
#        new_seq =  seq.update_from_sequence(res)
#        return new_seq
#
#    def viterbi_decode_log_raw(self,seq):
#        "Return only the prediction list and not a sequence"
#        node_potentials,edge_potentials = self.build_potentials_log(seq)
#        viterbi_path,_ = viterbi_log(node_potentials,edge_potentials)
#        res =  viterbi_path
#        return res
#
#
#    def viterbi_decode_corpus_log(self,seq_list):
#        predictions = []
#        for seq in seq_list:
#            predictions.append(self.viterbi_decode_log(seq))
#        return predictions
#
#    def evaluate_corpus(self,seq_list,predictions):
#        total = 0.0
#        correct = 0.0
#        for i,seq in enumerate(seq_list):
#            pred = predictions[i]
#            for i,y_hat in enumerate(pred.y):
#                if(seq.y[i] == y_hat):
#                    correct += 1
#                total += 1
#        return correct/total

    

#    def my_exp(self,number):
#        '''
#        Returns 1 in case of overflow
#        '''
#        try:
#            value = number #np.exp(number)
#        except:
#            print "Overflow computing exp"
#            print number
#            pdb.set_trace()
#            print self.parameters[edge_f_list]
#            value = 1
#        return value


#def ee(number):
#    try:
#        value = exp(number)
#    except:
#        print "Overflow computing exp"
#        print number
#        print self.parameters[edge_f_list]
#        value = 1
#    return value
#
#def text_exp3():
#    for i in xrange(100000):
#        ee(2)
#
#def test_exp():
#    for i in xrange(100000):
#        exp(2)
#
#def test_exp2():
#    for i in xrange(100000):
#        np.exp(2)
