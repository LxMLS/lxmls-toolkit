import numpy as np
import matplotlib.pyplot as plt
import sequence_classification_decoder as scd

import pdb

class SequenceClassifier():
    ''' Implements an abstract sequence classifier.'''

    def __init__(self, observation_labels, state_labels):
        '''Initialize a sequence classifier. observation_labels and
        state_labels are the sets of observations and states, respectively.
        They must be LabelDictionary objects.'''
        
        self.decoder = scd.SequenceClassificationDecoder()
        self.observation_labels = observation_labels
        self.state_labels = state_labels        
        self.trained = False
        
        
    def get_num_states(self):
        ''' Return the number of states.'''
        return len(self.state_labels)

        
    def get_num_observations(self):
        ''' Return the number of observations (e.g. word types).'''
        return len(self.observation_labels)
        

    def train_supervised(self, sequence_list):
        ''' Train a classifier in a supervised setting.'''
        raise NotImplementedError


    def compute_scores(self, sequence):
        ''' Compute emission and transition scores for the decoder.'''
        raise NotImplementedError
    

    def compute_posteriors(self, sequence):
        '''Compute the state and transition posteriors:
        - The state posteriors are the probability of each state
        occurring at each position given the sequence of observations.
        - The transition posteriors are the joint probability of two states
        in consecutive positions given the sequence of observations.
        Both quantities are computed via the forward-backward algorithm.'''

        num_states = self.get_num_states() # Number of states.
        length = len(sequence.x) # Length of the sequence.
        
#        pdb.set_trace()
        
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = \
            self.compute_scores(sequence)
            
        # Run the forward algorithm.
        log_likelihood, forward = self.decoder.run_forward(initial_scores,
                                                           transition_scores,
                                                           final_scores,
                                                           emission_scores)
#        print log_likelihood

        # Run the backward algorithm.
        log_likelihood, backward = self.decoder.run_backward(initial_scores,
                                                             transition_scores,
                                                             final_scores,
                                                             emission_scores)
#        print log_likelihood

        # Multiply the forward and backward variables to obtain the
        # state posteriors (sum in log-space).
        state_posteriors = np.zeros([length, num_states]) # State posteriors. 
        for pos in xrange(length):
            state_posteriors[pos,:] = forward[pos,:] + backward[pos,:]
            state_posteriors[pos,:] -= log_likelihood
 
        # Use the forward and backward variables along with the transition 
        # and emission scores to obtain the transition posteriors.
        transition_posteriors = np.zeros([length-1, num_states, num_states])
        for pos in xrange(length-1):
            for prev_state in xrange(num_states):
                for state in xrange(num_states):
                    transition_posteriors[pos, state, prev_state] = \
                        forward[pos, prev_state] + \
                        transition_scores[pos, state, prev_state] + \
                        emission_scores[pos+1, state] + \
                        backward[pos+1, state]
                    transition_posteriors[pos, state, prev_state] -= log_likelihood
                        
        state_posteriors = np.exp(state_posteriors)
        transition_posteriors = np.exp(transition_posteriors)
        
        return state_posteriors, transition_posteriors
        

    def posterior_decode(self, sequence):
        '''Compute the sequence of states that are individually the most
        probable, given the observations. This is done by maximizing
        the state posteriors, which are computed with the forward-backward
        algorithm.'''

        state_posteriors, _ = self.compute_posteriors(sequence)
        best_states =  np.argmax(state_posteriors, axis=1)
        predicted_sequence =  sequence.copy_sequence()
        predicted_sequence.y = best_states
        return predicted_sequence

    
    def posterior_decode_corpus(self, dataset):
        '''Run posterior_decode at corpus level.'''
        
        predictions = []
        for sequence in dataset.seq_list:
            predictions.append(self.posterior_decode(sequence))
        return predictions


    def viterbi_decode(self, sequence):
        '''Compute the most likely sequence of states given the observations,
        by running the Viterbi algorithm.'''

        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = \
            self.compute_scores(sequence)
            
        # Run the forward algorithm.
        best_states, total_score = self.decoder.run_viterbi(initial_scores,
                                                            transition_scores,
                                                            final_scores,
                                                            emission_scores)

        predicted_sequence =  sequence.copy_sequence()
        predicted_sequence.y = best_states
        return predicted_sequence, total_score



    def viterbi_decode_corpus(self, dataset):
        '''Run viterbi_decode at corpus level.'''

        predictions = []
        for sequence in dataset.seq_list:
            predicted_sequence, _ = self.viterbi_decode(sequence)
            predictions.append(predicted_sequence)
        return predictions


    def evaluate_corpus(self, dataset, predictions):
        '''Evaluate classification accuracy at corpus level, comparing with
        gold standard.'''
        total = 0.0
        correct = 0.0
        for i, sequence in enumerate(dataset.seq_list):
            pred = predictions[i]
            for i,y_hat in enumerate(pred.y):
                if(sequence.y[i] == y_hat):
                    correct += 1
                total += 1
        return correct/total

