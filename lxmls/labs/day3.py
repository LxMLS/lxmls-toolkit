import sys
sys.path.append("readers/" )
sys.path.append("sequences/" )


import structured_perceptron as spc
import crf_batch as crfc
import pos_corpus as pcc
import id_feature as idfc
import extended_feature as exfc


print "Perceptron Exercise"
posc = pcc.PostagCorpus("en",max_sent_len=15,train_sents=1000,dev_sents=200,test_sents=200)
id_f = idfc.IDFeatures(posc)
id_f.build_features()
sp = spc.StructuredPercetron(posc,id_f)
sp.nr_rounds = 20
sp.train_supervised(posc.train.seq_list)

pred_train = sp.viterbi_decode_corpus(posc.train.seq_list)
pred_dev = sp.viterbi_decode_corpus(posc.dev.seq_list)
pred_test = sp.viterbi_decode_corpus(posc.test.seq_list)

eval_train = sp.evaluate_corpus(posc.train.seq_list,pred_train)
eval_dev = sp.evaluate_corpus(posc.dev.seq_list,pred_dev)
eval_test = sp.evaluate_corpus(posc.test.seq_list,pred_test)

print "Structured Percetron - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)


ex_f = exfc.ExtendedFeatures(posc)
ex_f.build_features()
sp = spc.StructuredPercetron(posc,ex_f)
sp.nr_rounds = 20
sp.train_supervised(posc.train.seq_list)

pred_train = sp.viterbi_decode_corpus(posc.train.seq_list)
pred_dev = sp.viterbi_decode_corpus(posc.dev.seq_list)
pred_test = sp.viterbi_decode_corpus(posc.test.seq_list)

eval_train = sp.evaluate_corpus(posc.train.seq_list,pred_train)
eval_dev = sp.evaluate_corpus(posc.dev.seq_list,pred_dev)
eval_test = sp.evaluate_corpus(posc.test.seq_list,pred_test)

print "Structured Percetron - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)



print "CRF Exercise"
posc = pcc.PostagCorpus("en",max_sent_len=15,train_sents=1000,dev_sents=200,test_sents=200)
id_f = idfc.IDFeatures(posc)
id_f.build_features()


crf = crfc.CRF_batch(posc,id_f)
crf.train_supervised(posc.train.seq_list)

pred_train = crf.viterbi_decode_corpus(posc.train.seq_list)
pred_dev = crf.viterbi_decode_corpus(posc.dev.seq_list)
pred_test = crf.viterbi_decode_corpus(posc.test.seq_list)

eval_train = crf.evaluate_corpus(posc.train.seq_list,pred_train)
eval_dev = crf.evaluate_corpus(posc.dev.seq_list,pred_dev)
eval_test = crf.evaluate_corpus(posc.test.seq_list,pred_test)

print "CRF - ID Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)

posc = pcc.PostagCorpus("en",max_sent_len=15,train_sents=1000,dev_sents=200,test_sents=200)
ex_f = exfc.ExtendedFeatures(posc)
ex_f.build_features()


crf = crfc.CRF_batch(posc,ex_f)
crf.train_supervised(posc.train.seq_list)

pred_train = crf.viterbi_decode_corpus(posc.train.seq_list)
pred_dev = crf.viterbi_decode_corpus(posc.dev.seq_list)
pred_test = crf.viterbi_decode_corpus(posc.test.seq_list)

eval_train = crf.evaluate_corpus(posc.train.seq_list,pred_train)
eval_dev = crf.evaluate_corpus(posc.dev.seq_list,pred_dev)
eval_test = crf.evaluate_corpus(posc.test.seq_list,pred_test)

print "CRF - Extended Features Accuracy Train: %.3f Dev: %.3f Test: %.3f"%(eval_train,eval_dev,eval_test)
