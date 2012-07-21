
import numpy as np

from ..util.my_math_utils import *
from .viterbi import viterbi
from .viterbi_2 import viterbi_log
from .forward_backward import forward_backward


class DiscriminativeSequenceClassifier():


    def __init__(self,dataset,feature_class):
        self.trained = False
        self.feature_class = feature_class
        self.nr_states = len(dataset.int_to_tag)
        self.dataset = dataset
        self.feature_class = feature_class
        self.parameters = np.zeros([feature_class.nr_feats],dtype=float)
        

    def get_number_states(self):
        self.nr_states

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
    def build_potentials(self,sequence):
        nr_states = self.nr_states
        nr_pos = len(sequence.x)
        #Node potentials are indexed by state and position
        node_potentials = np.ones([nr_states,nr_pos],dtype=float)
        #Edge potentials are indexed by tag,next_tag, pos where pos is the position of tag
        edge_potentials = np.ones([nr_states,nr_states,nr_pos-1],dtype=float)
        ## We will assume that transition features do not depend on any X information so
        ## they are the same for all positions this will speed up the code but if the features
        ## change need to uncomment the code in the main loop and comment here
        for tag_id in xrange(nr_states):
            for prev_tag_id in xrange(nr_states):
                edge_f_list = self.feature_class.get_edge_features(sequence,1,tag_id,prev_tag_id)
                #print "Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
                #print edge_f_list
                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
                dot_w_f_edge = 0
                for fi in edge_f_list:
                    dot_w_f_edge += self.parameters[fi]
                edge_potentials[prev_tag_id,tag_id,:] = self.my_exp(dot_w_f_edge)
        #print "edge_potentials"
        #print edge_potentials
        ## Add first position
        for tag_id in xrange(nr_states):
            edge_f_list = self.feature_class.get_edge_features(sequence,0,tag_id,-1)
            dot_w_f_edge = 0
            for fi in edge_f_list:
                dot_w_f_edge += self.parameters[fi]
            node_potentials[tag_id,0] *= self.my_exp(dot_w_f_edge)
        #Add last position
        last_pos = len(sequence.x)
        for tag_id in xrange(nr_states):
                edge_f_list = self.feature_class.get_edge_features(sequence,last_pos,-1,tag_id)
                dot_w_f_edge = 0
                for fi in edge_f_list:
                    dot_w_f_edge += self.parameters[fi]
                node_potentials[tag_id,last_pos -1] *= self.my_exp(dot_w_f_edge)
        #print "edge_potentials after final"
        #print edge_potentials
        for pos,word_id in enumerate(sequence.x):
            for tag_id in xrange(nr_states):
                #f(t,y_t,X)
                node_f_list = self.feature_class.get_node_features(sequence,pos,tag_id)
                #print "Node list: pos:%i tag:%i"%(pos,tag_id)
                #print node_f_list
                ##w*f - since f is only 0/1 its just the sum of active features
                
                #dot_w_f_node = np.sum(self.parameters[node_f_list,:])
                dot_w_f_node = 0
                for fi in node_f_list:
                    dot_w_f_node += self.parameters[fi]
                node_potentials[tag_id,pos] *= self.my_exp(dot_w_f_node)
                
                ##Note this code is commented since we are assuming that transition features do not depende on X information
                # if(pos > 0):
                #     for prev_tag_id in all_tags:
                #         edge_f_list = self.feature_class.get_edge_features(sequence,pos,tag_id,prev_tag_id)
                #         dot_w_f_edge = np.sum(self.parameters[edge_f_list,:])
                #         edge_potentials[prev_tag_id,tag_id,pos-1] = np.exp(dot_w_f_edge)
        return node_potentials,edge_potentials


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
    def build_potentials_log(self,sequence):
        nr_states = self.nr_states
        nr_pos = len(sequence.x)
        node_potentials = np.zeros([nr_states,nr_pos],dtype=float)
        edge_potentials = np.zeros([nr_states,nr_states,nr_pos-1],dtype=float)


        ## We will assume that transition features do not depend on any X information so
        ## they are the same for all positions this will speed up the code but if the features
        ## change need to uncomment the code in the main loop and comment here
        for tag_id in xrange(nr_states):
            for prev_tag_id in xrange(nr_states):
                edge_f_list = self.feature_class.get_edge_features(sequence,1,tag_id,prev_tag_id)
                #print "Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
                #print edge_f_list
                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
                dot_w_f_edge = 0
                for fi in edge_f_list:
                    dot_w_f_edge += self.parameters[fi]
                edge_potentials[prev_tag_id,tag_id,:-1] = dot_w_f_edge
        #print "edge_potentials"
        #print edge_potentials
        ## Add last position
        last_pos = len(sequence.x)-1
        for tag_id in xrange(nr_states):
            for prev_tag_id in xrange(nr_states):
                #print "Final Edge list: tag:%i prev:%i"%(tag_id,prev_tag_id)
                #print edge_f_list
                edge_f_list = self.feature_class.get_edge_features(sequence,last_pos,tag_id,prev_tag_id)
                #dot_w_f_edge = np.sum(self.parameters[edge_f_list])
                dot_w_f_edge = 0
                for fi in edge_f_list:
                    dot_w_f_edge += self.parameters[fi]
                edge_potentials[prev_tag_id,tag_id,last_pos -1] = dot_w_f_edge
                
                    
        #print "edge_potentials after final"
        #print edge_potentials
        for pos,word_id in enumerate(sequence.x):
            for tag_id in xrange(nr_states):
                #f(t,y_t,X)
                node_f_list = self.feature_class.get_node_features(sequence,pos,tag_id)
                #print "Node list: pos:%i tag:%i"%(pos,tag_id)
                #print node_f_list
                ##w*f - since f is only 0/1 its just the sum of active features
                
                #dot_w_f_node = np.sum(self.parameters[node_f_list,:])
                dot_w_f_node = 0
                for fi in node_f_list:
                    dot_w_f_node += self.parameters[fi]
                node_potentials[tag_id,pos] = dot_w_f_node
                ##Note this code is commented since we are assuming that transition features do not depende on X information
                # if(pos > 0):
                #     for prev_tag_id in all_tags:
                #         edge_f_list = self.feature_class.get_edge_features(sequence,pos,tag_id,prev_tag_id)
                #         dot_w_f_edge = np.sum(self.parameters[edge_f_list,:])
                #         edge_potentials[prev_tag_id,tag_id,pos-1] = np.exp(dot_w_f_edge)
        return node_potentials,edge_potentials

    
    def forward_backward(self,seq):
        node_potentials,edge_potentials = self.build_potentials(seq)
        forward,backward = forward_backward(node_potentials,edge_potentials)
        #print sanity_check_forward_backward(forward,backward)
        return forward,backward
        

    ###############
    ## Returns the node posterios
    ####################
    def get_node_posteriors(self,seq):
        nr_states = self.nr_states
        node_potentials,edge_potentials = self.build_potentials(seq)
        forward,backward = forward_backward(node_potentials,edge_potentials)
        #print sanity_check_forward_backward(forward,backward)
        H,N = forward.shape
        likelihood = np.sum(forward[:,N-1])
        #print likelihood
        return self.get_node_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
        

    def get_node_posteriors_aux(self,seq,forward,backward,node_potentials,edge_potentials,likelihood):
        H,N = forward.shape
        posteriors = np.zeros([H,N],dtype=float)
        
        for pos in  xrange(N):
            for current_state in xrange(H):
                posteriors[current_state,pos] = forward[current_state,pos]*backward[current_state,pos]/likelihood
        return posteriors

    def get_edge_posteriors(self,seq):
        nr_states = self.nr_states
        node_potentials,edge_potentials = self.build_potentials(seq)
        forward,backward = forward_backward(node_potentials,edge_potentials)
        H,N = forward.shape
        likelihood = np.sum(forward[:,N-1])
        return self.get_edge_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
        
    def get_edge_posteriors_aux(self,seq,forward,backward,node_potentials,edge_potentials,likelihood):
        H,N = forward.shape
        edge_posteriors = np.zeros([H,H,N-1],dtype=float)
        for pos in xrange(N-1):
            for prev_state in xrange(H):
                for state in xrange(H):
                    edge_posteriors[prev_state,state,pos] = forward[prev_state,pos]*edge_potentials[prev_state,state,pos]*node_potentials[state,pos+1]*backward[state,pos+1]/likelihood 
        return edge_posteriors

    def posterior_decode(self,seq):
        posteriors = self.get_node_posteriors(seq)
        print posteriors
        res =  np.argmax(posteriors,axis=0)
        new_seq =  seq.copy_sequence()
        new_seq.y = res
        return new_seq
    
    def posterior_decode_corpus(self,seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.posterior_decode(seq))
        return predictions


    
    
    def viterbi_decode(self,seq):
        node_potentials,edge_potentials = self.build_potentials(seq)
        viterbi_path,_ = viterbi(node_potentials,edge_potentials)
        res =  viterbi_path
        new_seq =  seq.update_from_sequence(res)
        return new_seq



    def viterbi_decode_corpus(self,seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.viterbi_decode(seq))
        return predictions


    def viterbi_decode_log(self,seq):
        node_potentials,edge_potentials = self.build_potentials_log(seq)
        viterbi_path,_ = viterbi_log(node_potentials,edge_potentials)
        res =  viterbi_path
        new_seq =  seq.update_from_sequence(res)
        return new_seq

    def viterbi_decode_log_raw(self,seq):
        "Return only the prediction list and not a sequence"
        node_potentials,edge_potentials = self.build_potentials_log(seq)
        viterbi_path,_ = viterbi_log(node_potentials,edge_potentials)
        res =  viterbi_path
        return res


    def viterbi_decode_corpus_log(self,seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.viterbi_decode_log(seq))
        return predictions

    def evaluate_corpus(self,seq_list,predictions):
        total = 0.0
        correct = 0.0
        for i,seq in enumerate(seq_list):
            pred = predictions[i]
            for i,y_hat in enumerate(pred.y):
                if(seq.y[i] == y_hat):
                    correct += 1
                total += 1
        return correct/total

    

    def my_exp(self,number):
        '''
        Returns 1 in case of overflow
        '''
        try:
            value = exp(number)
        except:
            print "Overflow computing exp"
            print number
            print self.parameters[edge_f_list]
            value = 1
        return value


def ee(number):
    try:
        value = exp(number)
    except:
        print "Overflow computing exp"
        print number
        print self.parameters[edge_f_list]
        value = 1
    return value

def text_exp3():
    for i in xrange(100000):
        ee(2)

def test_exp():
    for i in xrange(100000):
        exp(2)

def test_exp2():
    for i in xrange(100000):
        np.exp(2)
