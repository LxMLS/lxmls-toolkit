import sys

sys.path.append('.')

import lxmls.sequences.crf_online as crfo
import lxmls.sequences.structured_perceptron as spc
import lxmls.readers.pos_corpus as pcc
import lxmls.sequences.id_feature as idfc
import lxmls.sequences.extended_feature as exfc

print "CRF Exercise"

corpus = pcc.PostagCorpus()
train_seq = corpus.read_sequence_list_conll("data/train-02-21.conll", max_sent_len=10, max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll("data/test-23.conll", max_sent_len=10, max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll("data/dev-22.conll", max_sent_len=10, max_nr_sent=1000)

feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()

crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
crf_online.num_epochs = 20
crf_online.train_supervised(train_seq)

pred_train = crf_online.viterbi_decode_corpus(train_seq)
pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
pred_test = crf_online.viterbi_decode_corpus(test_seq)
eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
eval_test = crf_online.evaluate_corpus(test_seq, pred_test)

print "CRF - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f" % (eval_train, eval_dev, eval_test)

feature_mapper = exfc.ExtendedFeatures(train_seq)
feature_mapper.build_features()

crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
crf_online.num_epochs = 20
crf_online.train_supervised(train_seq)

pred_train = crf_online.viterbi_decode_corpus(train_seq)
pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
pred_test = crf_online.viterbi_decode_corpus(test_seq)
eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
eval_test = crf_online.evaluate_corpus(test_seq, pred_test)

print "CRF - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f" % (eval_train, eval_dev, eval_test)

print "Perceptron Exercise"

feature_mapper = idfc.IDFeatures(train_seq)
feature_mapper.build_features()

sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)
eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

print "Structured Perceptron - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f" % (eval_train, eval_dev, eval_test)

feature_mapper = exfc.ExtendedFeatures(train_seq)
feature_mapper.build_features()

sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
sp.num_epochs = 20
sp.train_supervised(train_seq)

pred_train = sp.viterbi_decode_corpus(train_seq)
pred_dev = sp.viterbi_decode_corpus(dev_seq)
pred_test = sp.viterbi_decode_corpus(test_seq)
eval_train = sp.evaluate_corpus(train_seq, pred_train)
eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
eval_test = sp.evaluate_corpus(test_seq, pred_test)

print "Structured Perceptron - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f" % (eval_train, eval_dev, eval_test)


#
#
# import sequences.structured_perceptron as spc
# import sequences.crf_batch as crfc
# import sequences.crf_online as crfo
# import readers.pos_corpus as pcc
# import sequences.id_feature as idfc
# import sequences.extended_feature as exfc
# import pdb
#
# corpus = pcc.PostagCorpus()
# train_seq = corpus.read_sequence_list_conll("../data/train-02-21.conll",max_sent_len=10,max_nr_sent=1000)
# test_seq = corpus.read_sequence_list_conll("../data/test-23.conll",max_sent_len=10,max_nr_sent=1000)
# dev_seq = corpus.read_sequence_list_conll("../data/dev-22.conll",max_sent_len=10,max_nr_sent=1000)
# # corpus.add_sequence_list(train_seq)
# # id_f = idfc.IDFeatures(corpus)
# feature_mapper = idfc.IDFeatures(train_seq)
# feature_mapper.build_features()
#
#
# print "Perceptron Exercise"
#
# sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
# sp.num_epochs = 20
# sp.train_supervised(train_seq)
#
# pred_train = sp.viterbi_decode_corpus(train_seq)
# pred_dev = sp.viterbi_decode_corpus(dev_seq)
# pred_test = sp.viterbi_decode_corpus(test_seq)
#
# eval_train = sp.evaluate_corpus(train_seq, pred_train)
# eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
# eval_test = sp.evaluate_corpus(test_seq, pred_test)
#
# print "Structured Perceptron - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
#
# feature_mapper = exfc.ExtendedFeatures(train_seq)
# feature_mapper.build_features()
# sp = spc.StructuredPerceptron(corpus.word_dict, corpus.tag_dict, feature_mapper)
# sp.num_epochs = 20
# sp.train_supervised(train_seq)
#
# pred_train = sp.viterbi_decode_corpus(train_seq)
# pred_dev = sp.viterbi_decode_corpus(dev_seq)
# pred_test = sp.viterbi_decode_corpus(test_seq)
#
# eval_train = sp.evaluate_corpus(train_seq, pred_train)
# eval_dev = sp.evaluate_corpus(dev_seq, pred_dev)
# eval_test = sp.evaluate_corpus(test_seq, pred_test)
#
# print "Structured Perceptron - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
#
# # pdb.set_trace()
#
# # print "CRF Exercise"
#
# feature_mapper = idfc.IDFeatures(train_seq)
# feature_mapper.build_features()
#
# print "Online CRF Exercise"
#
# crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
# crf_online.num_epochs = 20
# # crf_online.initial_learning_rate = 10 #100 #1.0/crf_online.regularizer
# crf_online.train_supervised(train_seq)
#
# pred_train = crf_online.viterbi_decode_corpus(train_seq)
# pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
# pred_test = crf_online.viterbi_decode_corpus(test_seq)
#
# eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
# eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
# eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
#
# print "Online CRF - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
#
# crf = crfc.CRFBatch(corpus.word_dict, corpus.tag_dict, feature_mapper)
# crf.train_supervised(train_seq)
#
# pred_train = crf.viterbi_decode_corpus(train_seq)
# pred_dev = crf.viterbi_decode_corpus(dev_seq)
# pred_test = crf.viterbi_decode_corpus(test_seq)
#
# eval_train = crf.evaluate_corpus(train_seq, pred_train)
# eval_dev = crf.evaluate_corpus(dev_seq, pred_dev)
# eval_test = crf.evaluate_corpus(test_seq, pred_test)
#
# print "CRF - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
#
# # pdb.set_trace()
#
#
# feature_mapper = exfc.ExtendedFeatures(train_seq)
# feature_mapper.build_features()
#
#
#
# print "Online CRF Exercise"
#
# crf_online = crfo.CRFOnline(corpus.word_dict, corpus.tag_dict, feature_mapper)
# crf_online.num_epochs = 20
# # for eta in [1, 10, 100, 1000]:
# #    crf_online.initial_learning_rate = 10 #1.0/crf_online.regularizer
# crf_online.train_supervised(train_seq)
#
# pred_train = crf_online.viterbi_decode_corpus(train_seq)
# pred_dev = crf_online.viterbi_decode_corpus(dev_seq)
# pred_test = crf_online.viterbi_decode_corpus(test_seq)
#
# eval_train = crf_online.evaluate_corpus(train_seq, pred_train)
# eval_dev = crf_online.evaluate_corpus(dev_seq, pred_dev)
# eval_test = crf_online.evaluate_corpus(test_seq, pred_test)
#
# print "Online CRF - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
# # pdb.set_trace()
#
#
# crf = crfc.CRFBatch(corpus.word_dict, corpus.tag_dict, feature_mapper)
# crf.train_supervised(train_seq)
#
# pred_train = crf.viterbi_decode_corpus(train_seq)
# pred_dev = crf.viterbi_decode_corpus(dev_seq)
# pred_test = crf.viterbi_decode_corpus(test_seq)
#
# eval_train = crf.evaluate_corpus(train_seq, pred_train)
# eval_dev = crf.evaluate_corpus(dev_seq, pred_dev)
# eval_test = crf.evaluate_corpus(test_seq, pred_test)
#
# print "CRF - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
#
