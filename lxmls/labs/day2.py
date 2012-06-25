import sys
sys.path.append("readers/" )
sys.path.append("sequences/" )

import simple_sequence as ssr
import hmm as hmmc
import pos_corpus as pcc

#Exercise 4.1
print "Exercise 4.1"
simple = ssr.SimpleSequence()
simple.train
simple.test


#exercise 4.2
print "Exercise 4.2"
## See hmm.py file
hmm = hmmc.HMM(simple)
hmm.train_supervised(simple.train)
print "Init Probs"
print hmm.init_probs
print "Transition Probs"
print hmm.transition_probs
print "Final Probs"
print hmm.final_probs
print "Observation Probs"
print hmm.observation_probs

#exercise 4.3
print "Exercise 4.3"
## See hmm.py file
node_potentials,edge_potentials = hmm.build_potentials(simple.train.seq_list[0])
print "Node Potentials"
print node_potentials
print "Edge Potentials"
print edge_potentials

#exercise 4.4
print "Exercise 4.4"
## See forward_backward.py file
forward,backward =  hmm.forward_backward(simple.train.seq_list[0])
print "Likelihoods per position"
print hmm.sanity_check_fb(forward,backward)

#exercise 4.5
print "Exercise 4.5"
print "Node Posteriors"
node_posteriors = hmm.get_node_posteriors(simple.train.seq_list[0])
print node_posteriors

y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0"
print y_pred
print "Truth test 0"
print simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1"
print y_pred
print "Truth test 2"
print simple.test.seq_list[1]

#training with smoothing
hmm.train_supervised(simple.train,smoothing = 0.1)

y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0 with smoothing"
print y_pred
print "Truth test 0"
print simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1 with smoothing"
print y_pred
print "Truth test 2"
print simple.test.seq_list[1]

#exercise 4.6
print "Exercise 4.6"

y_pred = hmm.viterbi_decode(simple.test.seq_list[0])
print "Viterbi decoding Prediction test 1 with smoothing"
print y_pred
print "Truth test 2"
print simple.test.seq_list[0]

y_pred = hmm.viterbi_decode(simple.test.seq_list[1])
print "Viterbi decoding Prediction test 1 with smoothing"
print y_pred
print "Truth test 2"
print simple.test.seq_list[1]

#exercise 4.7
print "Exercise 4.7"
corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
corpus.add_sequence_list(train_seq) 
hmm = hmmc.HMM(corpus)
hmm.train_supervised(train_seq)



viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq)
posterior_pred_train = hmm.posterior_decode_corpus(train_seq)
eval_viterbi_train =   hmm.evaluate_corpus(train_seq,viterbi_pred_train)
eval_posterior_train = hmm.evaluate_corpus(train_seq,posterior_pred_train)
print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_train,eval_viterbi_train)

viterbi_pred_test = hmm.viterbi_decode_corpus(posc.test.seq_list)
posterior_pred_test = hmm.posterior_decode_corpus(posc.test.seq_list)
eval_viterbi_test =   hmm.evaluate_corpus(posc.test.seq_list,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(posc.test.seq_list,posterior_pred_test)
print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_test,eval_viterbi_test)


best_smothing = hmm.pick_best_smoothing(posc.train,posc.dev,[10,1,0.1,0])


hmm.train_supervised(posc.train,smoothing=best_smothing)
viterbi_pred_test = hmm.viterbi_decode_corpus(posc.test.seq_list)
posterior_pred_test = hmm.posterior_decode_corpus(posc.test.seq_list)
eval_viterbi_test =   hmm.evaluate_corpus(posc.test.seq_list,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(posc.test.seq_list,posterior_pred_test)
print "Best Smoothing %f --  Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(best_smothing,eval_posterior_test,eval_viterbi_test)

