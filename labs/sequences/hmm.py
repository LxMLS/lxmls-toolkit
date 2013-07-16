import numpy as np
import matplotlib.pyplot as plt
import sequence_classifier as sc
import sequences.confusion_matrix as cm
from log_domain import *

import pdb


class HMM(sc.SequenceClassifier):
    ''' Implements a first order HMM.'''

    def __init__(self, observation_labels, state_labels):
        '''Initialize an HMM. observation_labels and state_labels are the sets
        of observations and states, respectively. They are both LabelDictionary
        objects.'''
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)
        
        num_states = self.get_num_states()
        num_observations = self.get_num_observations()

        # Vector of probabilities for the initial states: P(state|START).
        self.initial_probs = np.zeros(num_states)
        
        # Matrix of transition probabilities: P(state|previous_state).
        # First index is the state, second index is the previous_state.
        self.transition_probs = np.zeros([num_states, num_states])
        
        # Vector of probabilities for the final states: P(STOP|state).
        self.final_probs = np.zeros(num_states)
                        
        # Matrix of emission probabilities. Entry (k,j) is probability
        # of observation k given state j. 
        self.emission_probs = np.zeros([num_observations, num_states])
        
        # Count tables.
        self.initial_counts = np.zeros(num_states)
        self.transition_counts = np.zeros([num_states, num_states])
        self.final_counts = np.zeros(num_states)
        self.emission_counts = np.zeros([num_observations, num_states])
        
                
    def train_EM(self, dataset, smoothing=0, num_epochs=10, evaluate=True):
#        pdb.set_trace()
        self.initialize_random()

        if evaluate:
            acc = self.evaluate_EM(dataset)
            print "Initial accuracy: %f"%(acc)
            
        for t in xrange(1, num_epochs):
            #E-Step
            total_log_likelihood = 0.0
            self.clear_counts(smoothing)
            for sequence in dataset.seq_list:
                # Compute scores given the observation sequence.
                initial_scores, transition_scores, final_scores, emission_scores = \
                    self.compute_scores(sequence)
                
                state_posteriors, transition_posteriors, log_likelihood = \
                    self.compute_posteriors(initial_scores,
                                            transition_scores,
                                            final_scores,
                                            emission_scores)
                self.update_counts(sequence, state_posteriors, transition_posteriors)
                total_log_likelihood += log_likelihood
            #self.model.sanity_check_counts(seq_list,smoothing=smoothing)
            print "Iter: %i Log Likelihood: %f"%(t, total_log_likelihood)
            #M-Step
            self.compute_parameters()
            if evaluate:
                 ### Evaluate accuracy at this iteration
                acc = self.evaluate_EM(dataset)
                print "Iter: %i Accuracy: %f"%(t,acc)
                
                
    def evaluate_EM(self, dataset):
        ### Evaluate accuracy at initial iteration
        pred = self.viterbi_decode_corpus(dataset)
#        pdb.set_trace()
        confusion_matrix = cm.build_confusion_matrix(dataset.seq_list, pred, 
                                                     self.get_num_states(), self.get_num_states())
        best = cm.get_best_assignment(confusion_matrix)
#        print best
        new_pred = []
        for i, sequence in enumerate(dataset.seq_list):
            pred_seq = pred[i]
            new_seq = pred_seq.copy_sequence()
            for j, y_hat in enumerate(new_seq.y):
                new_seq.y[j] = best[y_hat]
            new_pred.append(new_seq)
#        pdb.set_trace()
        acc = self.evaluate_corpus(dataset, new_pred)
        return acc
                

    def train_supervised(self, dataset, smoothing=0):
        ''' Train an HMM from a list of sequences containing observations
        and the gold states. This is just counting and normalizing.'''
        # Set all counts to zeros (optionally, smooth).
        self.clear_counts(smoothing)
        # Count occurrences of events.
        self.collect_counts_from_corpus(dataset)
        # Normalize to get probabilities.
        self.compute_parameters()

        
    def collect_counts_from_corpus(self, dataset):
        ''' Collects counts from a labeled corpus.'''
        for sequence in dataset.seq_list:
            # Take care of first position.
            self.initial_counts[sequence.y[0]] += 1
            self.emission_counts[sequence.x[0], sequence.y[0]] += 1

            # Take care of intermediate positions.
            for i, x in enumerate(sequence.x[1:]):
                y = sequence.y[i+1]
                y_prev = sequence.y[i]
                self.emission_counts[x, y] +=1
                self.transition_counts[y, y_prev] += 1

            # Take care of last position.
            self.final_counts[sequence.y[-1]] += 1


    ## Initializes the parameter randomnly
    def initialize_random(self):
        jitter = 1
        num_states = self.get_num_states()
        num_observations = self.get_num_observations()

        self.initial_counts.fill(1)
        self.initial_counts +=  jitter*np.random.rand(num_states)
        self.transition_counts.fill(1)
        self.transition_counts +=  jitter*np.random.rand(num_states, num_states)
        self.emission_counts.fill(1)
        self.emission_counts +=   jitter*np.random.rand(num_observations, num_states)
        self.final_counts.fill(1)
        self.final_counts +=  jitter*np.random.rand(num_states)
        self.compute_parameters()
        self.clear_counts()

        
    def clear_counts(self, smoothing = 0):
        ''' Clear all the count tables.'''
        self.initial_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.final_counts.fill(smoothing)
        self.emission_counts.fill(smoothing)


    def update_counts(self, sequence, state_posteriors, transition_posteriors):
        ''' Used in the E-step in EM.'''
        num_states = self.get_num_states() # Number of states.
        length = len(sequence.x) # Length of the sequence.

        ## Take care of initial probs
        for y in xrange(num_states):
            self.initial_counts[y] += state_posteriors[0, y]
        for pos in xrange(length):
            x = sequence.x[pos]
            for y in xrange(num_states):
                self.emission_counts[x, y] += state_posteriors[pos, y]
                if pos > 0:
                    for y_prev in xrange(num_states):
                        self.transition_counts[y, y_prev] += transition_posteriors[pos-1, y, y_prev]

        ##Final position
        for y in xrange(num_states):
            self.final_counts[y] += state_posteriors[length-1, y]
        
        
    def compute_parameters(self):
        ''' Estimate the HMM parameters by normalizing the counts.'''
        # Normalize the initial counts.
        sum_initial = np.sum(self.initial_counts)
        self.initial_probs = self.initial_counts / sum_initial
        
        # Normalize the transition counts and the final counts.
        sum_transition = np.sum(self.transition_counts, 0) + self.final_counts
        num_states = self.get_num_states()
        self.transition_probs = self.transition_counts / np.tile(sum_transition, [num_states, 1])
        self.final_probs = self.final_counts / sum_transition

        # Normalize the emission counts.
        sum_emission = np.sum(self.emission_counts, 0)
        num_observations = self.get_num_observations()
        self.emission_probs = self.emission_counts / np.tile(sum_emission, [num_observations, 1])

#    def update_counts(self,seq,posteriors):
#        node_posteriors,edge_posteriors = posteriors
#        H,N = node_posteriors.shape
#        ## Take care of initial probs
#        for y in xrange(H):
#            self.init_counts[y] += node_posteriors[y,0]
#            x = seq.x[0]
#            #print "x_%i=%i"%(0,x) 
#            self.observation_counts[x,y] += node_posteriors[y,0]
#        for pos in xrange(1,N-1):
#            x = seq.x[pos]
#            #print "x_%i=%i"%(pos,x) 
#            for y in xrange(H):
#                self.observation_counts[x,y] += node_posteriors[y,pos]
#                for y_next in xrange(H):
#                    ## pos-1 since edge_posteriors are indexed by prev_edge and not current edge
#                    self.transition_counts[y_next,y] += edge_posteriors[y,y_next,pos-1]
#
#        ##Final position
#        for y in xrange(H):
#            x = seq.x[N-1]
#            #print "x_%i=%i"%(N-1,x) 
#            self.observation_counts[x,y] += node_posteriors[y,N-1]
#            for y_next in xrange(H):
#                self.final_counts[y_next,y] += edge_posteriors[y,y_next,N-2]

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
    def sanity_check_counts(self, sequence_list, smoothing = 0):
        num_sentences = sequence_list.size()
        num_tokens = sequence_list.get_num_tokens() 
        
        print "Number of sentences: %i" % num_sentences
        print "Number of tokens: %i" % num_tokens
        
        sum_initials = np.sum(self.initial_counts)
        sum_transitions = np.sum(self.transition_counts) + np.sum(self.final_counts)
        sum_emissions = np.sum(self.emission_counts)
        
        # Compare:
        value = (num_sentences + smoothing * self.initial_counts.size)
        if (abs(sum_initials - value) > 0.001):
            print "Initial counts do not match: is - %f should be - %f"%(sum_initials, value)
        else:
            print "Initial counts match"
        value = num_tokens + smoothing * self.transition_counts.size
        if (abs(sum_transitions - value) > 0.001):
            print "Transition counts do not match: is - %f should be - %f"%(sum_transitions,value)
        else:
            print "Transition counts match"
        value = num_tokens +self.emission_counts.size*smoothing
        if (abs(sum_emissions - value) > 0.001):
            print "Emission counts do not match: is - %f should be - %f"%(sum_emissions,value)
        else:
            print "Emission Counts match"


    def compute_scores(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states]) + logzero()
        initial_scores = np.zeros(num_states) + logzero()
        transition_scores = np.zeros([length-1, num_states, num_states]) + logzero()
        final_scores = np.zeros(num_states) + logzero()

        # Initial position.
        initial_scores = safe_log(self.initial_probs)
        
        # Intermediate position.
        for pos in xrange(length):
            emission_scores[pos,:] = safe_log(self.emission_probs[sequence.x[pos], :])
            if pos > 0: 
                transition_scores[pos-1,:,:] = safe_log(self.transition_probs)

        # Final position.
        final_scores = safe_log(self.final_probs)

        return initial_scores, transition_scores, final_scores, emission_scores


#    def build_potentials(self,sequence):
#        nr_states = self.nr_states
#        nr_pos = len(sequence.x)
#        node_potentials = np.zeros([nr_states,nr_pos])
#        edge_potentials = np.zeros([nr_states,nr_states,nr_pos-1])
#        node_potentials[:,0] = self.observation_probs[sequence.x[0],:]*self.init_probs.transpose()
#        for pos in xrange(1,nr_pos):
#            edge_potentials[:,:,pos-1] = self.transition_probs[0:-1,:].transpose()
#            node_potentials[:,pos] = self.observation_probs[sequence.x[pos],:]
#
#        #Final position
#        node_potentials[:,nr_pos-1] *= self.transition_probs[-1,:].transpose()
#
#        return node_potentials,edge_potentials


#    def get_seq_prob(self,seq,node_potentials,edge_potentials):
#        nr_pos = len(seq.x)
#        #print "Node %i %i %.2f"%(0,seq.y[0],node_potentials[0,seq.y[0]])
#        value = node_potentials[0,seq.y[0]]
#        for pos in np.arange(1,nr_pos,1):
#            value *= node_potentials[seq.y[pos],pos]
#            #print "Node %i %i %.2f"%(pos,seq.y[pos],node_potentials[pos,seq.y[pos]])
#            value *= edge_potentials[seq.y[pos-1],seq.y[pos],pos-1]
#            #print "Edge Node %i %i %i %.2f"%(pos-1,seq.y[pos-1],seq.y[pos],edge_potentials[pos-1,seq.y[pos-1],seq.y[pos]])
#        return value
    

#    def forward_backward(self,seq):
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        forward,backward = forward_backward(node_potentials,edge_potentials)
##        sanity_check_forward_backward(forward,backward)
#        return forward,backward

#    def sanity_check_fb(self,forward,backward):
#        return sanity_check_forward_backward(forward,backward)


#    def compute_posteriors(self, sequence):
#        '''Compute the state and transition posteriors:
#        - The state posteriors are the probability of each state
#        occurring at each position given the sequence of observations.
#        - The transition posteriors are the joint probability of two states
#        in consecutive positions given the sequence of observations.
#        Both quantities are computed via the forward-backward algorithm.'''
#
#        num_states = self.get_num_states() # Number of states.
#        length = len(sequence.x) # Length of the sequence.
#        
#        # Compute scores given the observation sequence.
#        initial_scores, transition_scores, final_scores, emission_scores = \
#            self.compute_scores(sequence)
#            
#        # Run the forward algorithm.
#        likelihood, forward = run_forward(initial_scores,
#                                          transition_scores,
#                                          final_scores,
#                                          emission_scores)
#
#        # Run the backward algorithm.
#        likelihood, backward = run_backward(initial_scores,
#                                            transition_scores,
#                                            final_scores,
#                                            emission_scores)
#
#        # Multiply the forward and backward variables to obtain the
#        # state posteriors.
#        state_posteriors = np.zeros([length, num_states]) # State posteriors. 
#        for pos in  xrange(length):
#            state_posteriors[pos,:] = forward[pos,:] * backward[pos,:]
#            state_posteriors[pos,:] /= likelihood
# 
#        # Use the forward and backward variables along with the transition 
#        # and emission scores to obtain the transition posteriors.
#        transition_posteriors = np.zeros([length-1, num_states, num_states])
#        for pos in xrange(length-1):
#            for prev_state in xrange(num_states):
#                for state in xrange(num_states):
#                    transition_posteriors[pos, state, prev_state] = \
#                        forward[pos, prev_state] * \
#                        transition_scores[pos, state, prev_state] * \
#                        emission_scores[pos+1, state] * \
#                        backward[pos+1, state]
#                    transition_posteriors[pos, state, prev_state] /= likelihood
#                        
#        return state_posteriors, transition_posteriors
        

#    def get_node_posteriors_aux(self,seq,forward,backward,node_potentials,edge_potentials,likelihood):
#        H,N = forward.shape
#        posteriors = np.zeros([H,N],dtype=float)
#        
#        for pos in  xrange(N):
#            for current_state in xrange(H):
#                posteriors[current_state,pos] = forward[current_state,pos]*backward[current_state,pos]/likelihood
#        return posteriors

    
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

#    def get_posteriors(self,seq):
#        nr_states = self.nr_states
#        node_potentials,edge_potentials = self.build_potentials(seq)
#        forward,backward = forward_backward(node_potentials,edge_potentials)
#        #self.sanity_check_fb(forward,backward)
#        H,N = forward.shape
#        likelihood = np.sum(forward[:,N-1])
#        node_posteriors = self.get_node_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
#        edge_posteriors = self.get_edge_posteriors_aux(seq,forward,backward,node_potentials,edge_potentials,likelihood)
#        return [node_posteriors,edge_posteriors],likelihood 
    
        

#    def posterior_decode(self, sequence):
#        '''Compute the sequence of states that are individually the most
#        probable, given the observations. This is done by maximizing
#        the state posteriors, which are computed with the forward-backward
#        algorithm.'''
#
#        state_posteriors, _ = self.compute_posteriors(sequence)
#        best_states =  np.argmax(state_posteriors, axis=1)
#        predicted_sequence =  sequence.copy_sequence()
#        predicted_sequence.y = best_states
#        return predicted_sequence
#
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
#    def viterbi_decode(self, sequence):
#        '''Compute the most likely sequence of states given the observations,
#        by running the Viterbi algorithm.'''
#
#        # Compute scores given the observation sequence.
#        initial_scores, transition_scores, final_scores, emission_scores = \
#            self.compute_scores(sequence)
#            
#        # Run the forward algorithm.
#        best_states, total_score = run_viterbi(initial_scores,
#                                               transition_scores,
#                                               final_scores,
#                                               emission_scores)
#
#        predicted_sequence =  sequence.copy_sequence()
#        predicted_sequence.y = best_states
#        return predicted_sequence, total_score
#
#
#
#    def viterbi_decode_corpus(self,seq_list):
#        predictions = []
#        for seq in seq_list:
#            predicted_sequence, _ = self.viterbi_decode(seq)
#            predictions.append(predicted_sequence)
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

    ######
    # Plot the transition matrix for a given HMM
    ######
    def print_transition_matrix(self):
        print self.transition_probs
        cax = plt.imshow(self.transition_probs[0:-1,:], interpolation='nearest',aspect='auto')
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        print "Number os states %i"%self.get_num_states()
        print self.state_labels
        plt.xticks(np.arange(0, self.get_num_states()), self.state_labels.names, rotation=90)
        plt.yticks(np.arange(0, self.get_num_states()), self.state_labels.names)
        plt.show()

    def pick_best_smoothing(self,train,test,smooth_values):
        max_smooth = 0
        max_acc = 0
        for i in smooth_values:
               self.train_supervised(train, smoothing=i)
               viterbi_pred_train = self.viterbi_decode_corpus(train)
               posterior_pred_train = self.posterior_decode_corpus(train)
               eval_viterbi_train =   self.evaluate_corpus(train, viterbi_pred_train)
               eval_posterior_train = self.evaluate_corpus(train, posterior_pred_train)
               print "Smoothing %f --  Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_train,eval_viterbi_train)

               viterbi_pred_test = self.viterbi_decode_corpus(test)
               posterior_pred_test = self.posterior_decode_corpus(test)
               eval_viterbi_test =   self.evaluate_corpus(test, viterbi_pred_test)
               eval_posterior_test = self.evaluate_corpus(test, posterior_pred_test)
               print "Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(i,eval_posterior_test,eval_viterbi_test)
               if(eval_posterior_test > max_acc):
                   max_acc = eval_posterior_test
                   max_smooth = i
        return max_smooth


