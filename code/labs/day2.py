import sys
sys.path.append('.')
import readers.simple_sequence as ssr
import sequences.hmm as hmmc
import readers.pos_corpus as pcc
import sequences.confusion_matrix as cm
import pdb


print "------------"
print "Exercise 2.1"
print "------------"

simple = ssr.SimpleSequence()
print simple.train
print simple.test



print "------------"
print "Exercise 2.2"
print "------------"

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)

print "Initial Probabilities:"
print hmm.initial_probs
print "Transition Probabilities:"
print hmm.transition_probs
print "Final Probabilities:"
print hmm.final_probs
print "Emission Probabilities"
print hmm.emission_probs




print "------------"
print "Exercise 2.3"
print "------------"

initial_scores, transition_scores, final_scores, emission_scores = \
    hmm.compute_scores(simple.train.seq_list[0])
print initial_scores
print transition_scores
print final_scores
print emission_scores



print "------------"
print "Exercise 2.4"
print "------------"

import numpy as np

a = np.random.rand(10)
print np.log(sum(np.exp(a)))
print np.log(sum(np.exp(10*a)))
print np.log(sum(np.exp(100*a)))
print np.log(sum(np.exp(1000*a)))

from sequences.log_domain import *

print logsum(a)
print logsum(10*a)
print logsum(100*a)
print logsum(1000*a)





print "------------"
print "Exercise 2.5"
print "------------"

log_likelihood, forward = hmm.decoder.run_forward(initial_scores, transition_scores, final_scores, emission_scores)
print 'Log-Likelihood =', log_likelihood

log_likelihood, backward = hmm.decoder.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
print 'Log-Likelihood =', log_likelihood



print "------------"
print "Exercise 2.6"
print "------------"

initial_scores, transition_scores, final_scores, emission_scores = \
    hmm.compute_scores(simple.train.seq_list[0])
state_posteriors, _, _ = hmm.compute_posteriors(initial_scores,
                                                transition_scores,
                                                final_scores,
                                                emission_scores)
print state_posteriors




print "------------"
print "Exercise 2.7"
print "------------"

y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "Prediction test 0", y_pred
print "Truth test 0", simple.test.seq_list[0]

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "Prediction test 1", y_pred
print "Truth test 1", simple.test.seq_list[1]

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




print "------------"
print "Exercise 2.8"
print "------------"

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






print "------------"
print "Exercise 2.9"
print "------------"

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("../data/test-23.conll",max_sent_len=15,max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll",max_sent_len=15,max_nr_sent=1000)
hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_supervised(train_seq)

viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq)
posterior_pred_train = hmm.posterior_decode_corpus(train_seq)
eval_viterbi_train =   hmm.evaluate_corpus(train_seq, viterbi_pred_train)
eval_posterior_train = hmm.evaluate_corpus(train_seq, posterior_pred_train)
print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_train,eval_viterbi_train)

viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test =   hmm.evaluate_corpus(test_seq,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq,posterior_pred_test)
print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_test,eval_viterbi_test)

best_smothing = hmm.pick_best_smoothing(train_seq, dev_seq, [10,1,0.1,0])


hmm.train_supervised(train_seq, smoothing=best_smothing)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test =   hmm.evaluate_corpus(test_seq, viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq, posterior_pred_test)
print "Best Smoothing %f --  Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(best_smothing,eval_posterior_test,eval_viterbi_test)

confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, viterbi_pred_test, 
                                             len(corpus.tag_dict), hmm.get_num_states())
cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict, 
                            range(hmm.get_num_states()), 'Confusion matrix')



print "------------"
print "Exercise 2.10"
print "------------"

# Train with EM.
hmm.train_EM(train_seq, 0.1, 20, evaluate=True)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test =   hmm.evaluate_corpus(test_seq, viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq, posterior_pred_test)


confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, viterbi_pred_test, 
                                             len(corpus.tag_dict), hmm.get_num_states())
cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict, 
                            xrange(hmm.get_num_states()), 'Confusion matrix')


