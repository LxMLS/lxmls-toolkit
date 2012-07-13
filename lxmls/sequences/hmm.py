import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("util/" )

from my_math_utils import *
from viterbi import viterbi
from forward_backward import forward_backward,sanity_check_forward_backward



class HMM():
    ''' Implements a first order HMM'''

    def __init__(self,dataset,nr_states=-1):
        if(nr_states == -1):
            self.nr_states = len(dataset.int_to_tag)
        else:
            self.nr_states = nr_states
        self.nr_types = len(dataset.int_to_word)
        self.dataset = dataset
        self.trained = False
        self.init_probs = np.zeros([self.nr_states,1],dtype=float)
        self.transition_probs = np.zeros([self.nr_states+1,self.nr_states],dtype=float)
        self.observation_probs = np.zeros([self.nr_types,self.nr_states],dtype=float)
        ## Model counts tables
        # c(s_1 = s)
        self.init_counts = np.zeros([self.nr_states,1],dtype=float)
        # c(s_t = s | s_t-1 = q)
        # Includes extra row for the stopping probability (stop symbol)
        self.transition_counts = np.zeros([self.nr_states+1,self.nr_states],dtype=float)
        # c(o_t = v | s_t = q)
        self.observation_counts = np.zeros([self.nr_types,self.nr_states],dtype=float)
        
    def get_number_states(self):
        self.nr_states


    def train_supervised(self,sequence_list, smoothing=0):
        if(len(self.dataset.int_to_tag) != self.nr_states):
            print "Cannot train supervised models with number of states different than true pos tags"
            return
        
        nr_types = len(sequence_list.x_dict)
        nr_states = len(sequence_list.y_dict)
        ## Sets all counts to zeros
        self.clear_counts(smoothing)
        self.collect_counts_from_corpus(sequence_list)
        self.update_params()
        
    def collect_counts_from_corpus(self,sequence_list):
        ''' Collects counts from a labeled corpus'''
        for sequence in sequence_list.seq_list:
            len_x = len(sequence.x)
            #Take care of first position
            self.init_counts[sequence.y[0],0] +=1
            self.observation_counts[sequence.x[0],sequence.y[0]] +=1
            idx = 0
            for i,x in enumerate(sequence.x[1:]):
                idx = i+1
                y = sequence.y[idx]
                y_prev = sequence.y[idx-1]
                self.observation_counts[x,y] +=1
                self.transition_counts[y,y_prev] += 1
            ##Take care of last position
            y = sequence.y[idx]
            self.transition_counts[-1,y] += 1

    ## Initializes the parameter randomnly
    def initialize_random(self):
        jitter = 1
        self.init_counts.fill(1)
        self.init_counts +=  jitter*np.random.rand(self.init_counts.shape[0],self.init_counts.shape[1])
        self.transition_counts.fill(1)
        self.transition_counts +=  jitter*np.random.rand(self.transition_counts.shape[0],self.transition_counts.shape[1])
        self.observation_counts.fill(1)
        self.observation_counts +=   jitter*np.random.rand(self.observation_counts.shape[0],self.observation_counts.shape[1])
        self.update_params()
        self.clear_counts()
        
    def clear_counts(self,smoothing = 0):
        self.init_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.observation_counts.fill(smoothing)

    def update_params(self):
        # Normalize
        self.init_probs = normalize_array(self.init_counts)
        self.transition_probs = normalize_array(self.transition_counts)
        self.observation_probs = normalize_array(self.observation_counts)

    def update_counts(self,seq,posteriors):
        node_posteriors,edge_posteriors = posteriors
        H,N = node_posteriors.shape
        ## Take care of initial probs
        for y in xrange(H):
            self.init_counts[y] += node_posteriors[y,0]
            x = seq.x[0]
            #print "x_%i=%i"%(0,x) 
            self.observation_counts[x,y] += node_posteriors[y,0]
        for pos in xrange(1,N-1):
            x = seq.x[pos]
            #print "x_%i=%i"%(pos,x) 
            for y in xrange(H):
                self.observation_counts[x,y] += node_posteriors[y,pos]
                for y_next in xrange(H):
                    ## pos-1 since edge_posteriors are indexed by prev_edge and not current edge
                    self.transition_counts[y_next,y] += edge_posteriors[y,y_next,pos-1]

        ##Final position
        for y in xrange(H):
            x = seq.x[N-1]
            #print "x_%i=%i"%(N-1,x) 
            self.observation_counts[x,y] += node_posteriors[y,N-1]
            for y_next in xrange(H):
                self.final_counts[y_next,y] += edge_posteriors[y,y_next,N-2]

        #print "Observation counts"
        #print self.observation_counts

    #####
    # Check if the collected counts make sense
    # Init Counts - Should sum to the number of sentences
    # Transition Counts  - Should sum to number of tokens - number of sentences
    # Observation counts - Should sum to the number of tokens
    #
    # Seq_list should be the same used for train.
    # NOTE: If you use smoothing when trainig you have to account for that when comparing the values
    ######
    def sanity_check_counts(self,seq_list,smoothing = 0):
        nr_sentences = len(seq_list.seq_list)
        nr_tokens = sum(map(lambda seq: len(seq.x), seq_list.seq_list))
        print "Nr_sentence: %i"%nr_sentences
        print "Nr_tokens: %i"%nr_tokens
        sum_init = np.sum(self.init_counts)
        sum_transition = np.sum(self.transition_counts)
        sum_observations = np.sum(self.observation_counts)
        ##Compare
        value = (nr_sentences +smoothing*self.init_counts.size)
        if(abs(sum_init - value) > 0.001):
            print "Init counts do not match: is - %f should be - %f"%(sum_init,value)
        else:
            print "Init Counts match"
        value = nr_tokens + smoothing*self.transition_counts.size
        if(abs(sum_transition - value) > 0.001):
            print "Transition counts do not match: is - %f should be - %f"%(sum_transition,value)
        else:
            print "Transition Counts match"
        value = nr_tokens +self.observation_counts.size*smoothing
        if(abs(sum_observations - value) > 0.001):
            print "Observations counts do not match: is - %f should be - %f"%(sum_observations,value)
        else:
            print "Observations Counts match"


    def build_potentials(self,sequence):
        nr_states = self.nr_states
        nr_pos = len(sequence.x)
        node_potentials = np.zeros([nr_states,nr_pos])
        edge_potentials = np.zeros([nr_states,nr_states,nr_pos-1])
        node_potentials[:,0] = self.observation_probs[sequence.x[0],:]*self.init_probs.transpose()
        for pos in xrange(1,nr_pos):
            edge_potentials[:,:,pos-1] = self.transition_probs[0:-1,:].transpose()
            node_potentials[:,pos] = self.observation_probs[sequence.x[pos],:]

        #Final position
        node_potentials[:,nr_pos-1] *= self.transition_probs[-1,:].transpose()

        return node_potentials,edge_potentials


    def get_seq_prob(self,seq,node_potentials,edge_potentials):
        nr_pos = len(seq.x)
        #print "Node %i %i %.2f"%(0,seq.y[0],node_potentials[0,seq.y[0]])
        value = node_potentials[0,seq.y[0]]
        for pos in np.arange(1,nr_pos,1):
            value *= node_potentials[seq.y[pos],pos]
            #print "Node %i %i %.2f"%(pos,seq.y[pos],node_potentials[pos,seq.y[pos]])
            value *= edge_potentials[seq.y[pos-1],seq.y[pos],pos-1]
            #print "Edge Node %i %i %i %.2f"%(pos-1,seq.y[pos-1],seq.y[pos],edge_potentials[pos-1,seq.y[pos-1],seq.y[pos]])
        return value
    

    def forward_backward(self,seq):
        node_potentials,edge_potentials = self.build_potentials(seq)
        forward,backward = forward_backward(node_potentials,edge_potentials)
        sanity_check_forward_backward(forward,backward)
        return forward,backward

    def sanity_check_fb(self,forward,backward):
        return sanity_check_forward_backward(forward,backward)

    
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

    def get_posteriors(self,seq):
        nr_states = self.nr_states
        node_potentials,edge_potentials = self.build_potentials(seq)
        forward,backward = forward_backward(node_potentials,edge_potentials)
        #self.sanity_check_fb(forward,backward)
        H,N = forward.shape
        likelihood = np.sum(forward[:,N-1])
        node_posteriors = self.get_node_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
        edge_posteriors = self.get_edge_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
        return [node_posteriors,edge_posteriors],likelihood 
    
        

    def posterior_decode(self,seq):
        posteriors = self.get_node_posteriors(seq)
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
        new_seq =  seq.copy_sequence()
        new_seq.y = res
        return new_seq



    def viterbi_decode_corpus(self,seq_list):
        predictions = []
        for seq in seq_list:
            predictions.append(self.viterbi_decode(seq))
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

    ######
    # Plot the transition matrix for a given HMM
    ######
    def print_transition_matrix(self):
        cax = plt.imshow(self.transition_probs[0:-1,:], interpolation='nearest',aspect='auto')
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        plt.xticks(np.arange(0,len(self.nr_states)),np.arange(self.nr_staets),rotation=90)
        plt.yticks(np.arange(0,len(self.nr_states)),np.arange(self.nr_staets))


    def pick_best_smoothing(self,train,test,smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
               self.train_supervised(train,smoothing=i)
               viterbi_pred_train = self.viterbi_decode_corpus(train.seq_list)
               posterior_pred_train = self.posterior_decode_corpus(train.seq_list)
               eval_viterbi_train =   self.evaluate_corpus(train.seq_list,viterbi_pred_train)
               eval_posterior_train = self.evaluate_corpus(train.seq_list,posterior_pred_train)
               print "Smoothing %f --  Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_train,eval_viterbi_train)

               viterbi_pred_test = self.viterbi_decode_corpus(test.seq_list)
               posterior_pred_test = self.posterior_decode_corpus(test.seq_list)
               eval_viterbi_test =   self.evaluate_corpus(test.seq_list,viterbi_pred_test)
               eval_posterior_test = self.evaluate_corpus(test.seq_list,posterior_pred_test)
               print "Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_test,eval_viterbi_test)
               if(eval_posterior_test > max_acc):
                   max_acc = eval_posterior_test
                   max_smooth = i
        return max_smooth


