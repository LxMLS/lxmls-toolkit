import sys
sys.path.append('.')
import readers.simple_sequence as ssr
import sequences.hmm as hmmc
import readers.pos_corpus as pcc
import pdb




#Exercise 2.1
print "Exercise 2.1"
simple = ssr.SimpleSequence()
simple.train
simple.test


#exercise 2.2
print "Exercise 2.2"
## See hmm.py file
hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
hmm.sanity_check_counts(simple.train)
print "Initial Probabilities:"
print hmm.initial_probs
print "Transition Probabilities:"
print hmm.transition_probs
print "Final Probabilities:"
print hmm.final_probs
print "Emission Probabilities"
print hmm.emission_probs

##exercise 2.3
#print "Exercise 2.3"
### See hmm.py file
#node_potentials,edge_potentials = hmm.build_potentials(simple.train.seq_list[0])
#print "Node Potentials"
#print node_potentials
#print "Edge Potentials"
#print edge_potentials


#exercise 2.4
print "Exercise 2.4"
## See forward_backward.py file
#forward,backward =  hmm.forward_backward(simple.train.seq_list[0])
#print "Likelihoods per position"
#print hmm.sanity_check_fb(forward,backward)

#exercise 2.5
print "Exercise 2.5"
print "State Posteriors"
state_posteriors, _ = hmm.compute_posteriors(simple.train.seq_list[0])
print state_posteriors



y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0"
print y_pred
print "Truth test 0"
print simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1"
print y_pred
print "Truth test 1"
print simple.test.seq_list[1]


#training with smoothing
hmm.train_supervised(simple.train, smoothing = 0.1)

y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0 with smoothing"
print y_pred
print "Truth test 0"
print simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1 with smoothing"
print y_pred
print "Truth test 1"
print simple.test.seq_list[1]

#exercise 2.6
print "Exercise 2.6"

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])
print "Viterbi decoding Prediction test 0 with smoothing"
print y_pred, score
print "Truth test 0"
print simple.test.seq_list[0]

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[1])
print "Viterbi decoding Prediction test 1 with smoothing"
print y_pred, score
print "Truth test 1"
print simple.test.seq_list[1]

pdb.set_trace()

# RIGHT NOW, WHOEVER COMPLETED VITERBI KNOWS ONLY THAT THEIR PREDICTED SEQUENCES MATCH 
# THE ONES IN THE GUIDE. BUT THESE DOES NOT MEAN THAT THERE IS NOT A BUG 
# SOMEWHERE.
# SUGGESTION: MAKE THEM OUTPUT THE PROBABILITY OF THE BEST SEQUENCE.
# FOR THIS, THEY NEED TO DIVIDE THE BEST SCORE (RETURNED BY VITERBI) 
# BY THE LIKELIHOOD (BY RUNNING THE FORWARD ALGO)


#exercise 2.7
print "Exercise 2.7"
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("../data/test-23.conll",max_sent_len=15,max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll",max_sent_len=15,max_nr_sent=1000)
#corpus.add_sequence_list(train_seq) 
hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_supervised(train_seq)

viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq.seq_list)
posterior_pred_train = hmm.posterior_decode_corpus(train_seq.seq_list)
eval_viterbi_train =   hmm.evaluate_corpus(train_seq.seq_list,viterbi_pred_train)
eval_posterior_train = hmm.evaluate_corpus(train_seq.seq_list,posterior_pred_train)
print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_train,eval_viterbi_train)




viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq.seq_list)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq.seq_list)
eval_viterbi_test =   hmm.evaluate_corpus(test_seq.seq_list,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq.seq_list,posterior_pred_test)
print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_test,eval_viterbi_test)


best_smothing = hmm.pick_best_smoothing(train_seq,dev_seq,[10,1,0.1,0])


hmm.train_supervised(train_seq,smoothing=best_smothing)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq.seq_list)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq.seq_list)
eval_viterbi_test =   hmm.evaluate_corpus(test_seq.seq_list,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq.seq_list,posterior_pred_test)
print "Best Smoothing %f --  Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(best_smothing,eval_posterior_test,eval_viterbi_test)

