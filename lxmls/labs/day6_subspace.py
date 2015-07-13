#!/usr/bin/python
import shutil
import lxmls.deep_learning.SemEvalTwitter as ST
import lxmls.deep_learning.sgd as sgd
import numpy as np
import theano
import theano.tensor as T
import cPickle
import sys
import os
import lxmls.deep_learning.dnn as dnn
# import Print
#
# from Fmeasure import FmesSemEval

# DEBUGGING
from ipdb import set_trace


####################################
#           CONFIG 
####################################

# FLAG
DO_TRAIN        = 1
COMPILED_UPDATE = True 
TEST_ALL_MODELS = False 
MAKE_RUN        = False 
TEST_2015       = True # Avoid using this often until paper is shipped

# TRAIN CONFIGURATION
n_iter  = 8
lrate   = np.array(0.01).astype(theano.config.floatX)

# READ SEMEVAL DATA
sr = ST.SemEvalReader(one_hot=False)
dev_x, dev_y     = sr.dev
train_x, train_y = sr.train

# t2015_x, t2015_y = sr.tweets2015
# t2014_x, t2014_y = sr.tweets2014
# t2013_x, t2013_y = sr.tweets2013

#convert the data to play nice with Theano
# dev_x   = dev_x.astype(theano.config.floatX)
# dev_y   =  dev_y.astype('int32')
# train_x = train_x.astype(theano.config.floatX)
# train_y = train_y.astype('int32')
train_y = [np.array(ty).astype('int32')[None] for ty in train_y]
dev_y = [np.array(dy).astype('int32')[None] for dy in dev_y]
# t2015_x = t2015_x.astype(theano.config.floatX)
# t2015_y = t2015_y.astype('int32')
# t2014_x = t2014_x.astype(theano.config.floatX)
# t2014_y = t2014_y.astype('int32')
# t2013_x = t2013_x.astype(theano.config.floatX)
# t2013_y = t2013_y.astype('int32')
emb_path = 'DATA/twitter/features/E.pkl'



# SUBSPACE EMBEDDING
subsize    = 10
hidden     = False 
model_path = ('models/TSA_%d_iter%d_lrate001.pkl' % 
              (subsize, n_iter))
bsize    = 1
n_batch  = len(train_x)/bsize  + 1
# Create or load model
if DO_TRAIN:
    nn  = dnn.embSubMLP(emb_path, n_h1=subsize, hidden=hidden)
else:
    nn = dnn.embSubMLP(None, model_file=model_path)



####################################
#           SGD TRAIN 
####################################

if DO_TRAIN:

    print "Training %s" % model_path
    # RESHAPE TRAIN DATA AS A SINGLE NUMPY ARRAY
    # Start and end indices
    lens = np.array([len(tr) for tr in train_x]).astype('int32')
    st   = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype('int32')
    ed   = (st + lens).astype('int32')
    x    = np.zeros((ed[-1], 1))
    for i, ins_x in enumerate(train_x):        
        x[st[i]:ed[i]] = ins_x[:, None].astype('int32') 

    # FUNCTION FOR BATCH UPDATE
    # Train data and instance start and ends
    x  = theano.shared(x.astype('int32'), borrow=True) 
    y  = theano.shared(np.array(train_y).astype('int32'), borrow=True)
    st = theano.shared(st, borrow=True)
    ed = theano.shared(ed, borrow=True)
    # Update rule
    updates = [(pr, pr-lrate*gr) for pr, gr in zip(nn.params, nn.nablas)]
    # Mini-batch
    i  = T.lscalar()
    givens={ nn.z0 : x[st[i]:ed[i], 0],
             nn.y  : y[i] }
    train_batch = theano.function(inputs=[i], outputs=nn.F, updates=updates, givens=givens)


    sgd.SGD_train(nn, n_iter, batch_up=train_batch, n_batch=n_batch, devel_set=(dev_x,dev_y))

    # Epoch loop
    last_cr  = None
    best_cr  = [0, 0]
    for i in np.arange(n_iter):

        # Training Epoch 
        if COMPILED_UPDATE:
            # COMPILED BATCH UPDATE 
            p_train = 0 
            for j in np.arange(len(train_x)).astype('int32'): 
                p_train += train_batch(j) 
                # DEBUG
                if np.any(np.isnan(nn.params[0].get_value())):
                    set_trace()
                    print ""
                # DEBUG

                # INFO
                if not (j % 100):
                    sys.stdout.write("\rTraining %d/%d" % (j+1, len(train_x)))
                    sys.stdout.flush()   
    
        else:
            # MANUAL BATCH UPDATE 
            for j, x, y in zip(np.arange(len(train_y)), train_x, train_y): 
                for pr, gr in zip(nn.params, nn.grads): 
                    nablaF = np.asarray(gr(x[0], y))
                    # DEBUG 
                    if np.any(np.isnan(nablaF)):
                        pass
                    # DEBUG 
                    else:
                        pr.set_value(pr.get_value() - lrate*nablaF)
                    # DEBUG 
                    if np.any(np.isnan(pr.get_value())):
                        set_trace()
                        print ""
                    # DEBUG 

                # INFO
                if not (j % 100):
                    sys.stdout.write("\rTraining %d/%d" % (j+1, len(train_x)))
                    sys.stdout.flush()   

        raise Exception, ("That is enough!")
        # Evaluation
        cr      = 0.
        mapp    = np.array([ 1, 2, 0])
        ConfMat = np.zeros((3, 3))
        for j, x, y in zip(np.arange(len(dev_x)), dev_x, dev_y):
            # Prediction
            p_y   = nn.forward(x[0])
            hat_y = np.argmax(p_y)
            # Confusion matrix
            ConfMat[mapp[y[0]], mapp[hat_y]] += 1
            # Accuracy
            cr    = (cr*j + (hat_y == y[0]).astype(float))/(j+1)
            # INFO
            sys.stdout.write("\rDevel %d/%d            " % (j+1, len(dev_x)))
            sys.stdout.flush()   
        #
        Fm = FmesSemEval(ConfMat)
    
        # INFO
        if last_cr:
            # Keep bet model
            if best_cr[0] < cr:
                best_cr = [cr, i+1]
            delta_cr = cr - last_cr
            if delta_cr >= 0:
                print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[32m+%2.5f\033[0m (Fm %2.5f%%)" % 
                       (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
            else: 
                print ("\rEpoch %2d/%2d: Acc %2.5f%% \033[31m%2.5f\033[0m (Fm %2.5f%%)" % 
                       (i+1, n_iter, cr*100, delta_cr*100, Fm*100))
        else:
            print "\rEpoch %2d/%2d: %2.5f (Fm %2.5f%%)" % (i+1, n_iter, cr*100,
                                                           Fm*100)
            best_cr = [cr, i+1]
        last_cr = cr

#        print p_train
    
        # SAVE MODEL
        tmp_model_path = model_path.replace('.pkl','.%d.pkl' % (i+1))
        nn.save(tmp_model_path)

    # Store best model with the original model name
    tmp_model_path = model_path.replace('.pkl','.%d.pkl' % best_cr[1])
    print "Best model %s -> %s\nDev %2.5f %%" % (tmp_model_path, 
                                                 model_path, best_cr[0]*100)
    shutil.copy(tmp_model_path, model_path)

if MAKE_RUN:

    # Re-load best model
    if 1:
        nn = dnn.embSubMLP(None, model_file=model_path)
    else:
        nn = dnn.embLogLinear(None, model_file=model_path)

    # SEMEVAL 2015 OFFICIAL TWEETS
    print "Loading %s" % official_data
    with open(official_data, 'rb') as fid:
        [word2idx_eval, idx2word_eval, eval_x, eval_y] = cPickle.load(fid)

    with open(predictions, 'w') as fid:
        for j, x, y in zip(np.arange(len(eval_x)), eval_x, eval_y):
        # for j, x, y in zip(np.arange(100), eval_x, eval_y):
            hat_y = np.argmax(nn.forward(x[0]))
            label = None
            if hat_y == 0:
                label = "positive"
            elif hat_y == 1:
                label = "negative"
            elif hat_y == 2:
                label = "neutral"

            fid.write("NA\t%s\t%s\n" % (y, label) )
    print "Done"
else:
    ####################################
    #              TEST
    ####################################

    if TEST_ALL_MODELS:
       # Store all models fitting that condition
       i = 1
       model_path_list = [] 
       while os.path.isfile(model_path.replace('.pkl','.%d.pkl' % i)):
           model_path_list.append(model_path.replace('.pkl','.%d.pkl' % i))
           i += 1
    else:
        model_path_list = [model_path]


    for test_model in model_path_list:

        # Re-load best model
        if 1:
            print "Testing with %s" % test_model
            nn = dnn.embSubMLP(None, model_file=test_model)
        else:
            print "Testing with %s" % test_model
            nn = dnn.embLogLinear(None, model_file=test_model)
    
        if TEST_2015: 
            print "Testing on Tweets2015"
            # SEMEVAL TRAIN TWEETS
            print "Loading %s" % official_data
            with open(official_data, 'rb') as fid:
                [word2idx_eval, idx2word_eval, eval_x, eval_y] = cPickle.load(fid)
    
            # this not needed anymore (i am doing this conversion when 
            # extracting the features)
            # eval_x = [tx.astype('int32') for tx in eval_x]
            eval_y = [np.array(ty).astype('int32')[None] for ty in eval_y]
    
            # Evaluation
            # mapping for the confusion magtrix
            # yo: 0 pos, 1 neg, 2, neu
            #     0 neu, 1 pos, 2, neg
            mapp    = np.array([ 1, 2, 0])
            ConfMat = np.zeros((3, 3))
            cr      = 0.
            for j, x, y in zip(np.arange(len(eval_x)), eval_x, eval_y):
                hat_y = np.argmax(nn.forward(x[0]))
                ConfMat[mapp[y[0]], mapp[hat_y]] += 1
                cr    = (cr*j + (hat_y == y[0]).astype(float))/(j+1)
                # INFO
                sys.stdout.write("\rEvaluation %d/%d" % (j, len(eval_x)))
                sys.stdout.flush()
            print " Accuracy %2.5f %% " % (cr*100),
            print "Polar F-measure average: %0.4f" % Print.posNegFmeasure(ConfMat)
            print ""

	print "Testing on Tweets2014"
        # SEMEVAL TRAIN TWEETS
        print "Loading %s" % test_data_14
        with open(test_data_14, 'rb') as fid:
            [word2idx_eval, idx2word_eval, eval_x, eval_y] = cPickle.load(fid)
    
        #this not needed anymore (i am doing this conversion when extracting the features)
        # eval_x = [tx.astype('int32') for tx in eval_x]
        eval_y = [np.array(ty).astype('int32')[None] for ty in eval_y]
    
        # Evaluation
        # mapping for the confusion magtrix
        # yo: 0 pos, 1 neg, 2, neu
        #     0 neu, 1 pos, 2, neg
        mapp    = np.array([ 1, 2, 0])
        ConfMat = np.zeros((3, 3))
        cr      = 0.
        for j, x, y in zip(np.arange(len(eval_x)), eval_x, eval_y):
            hat_y = np.argmax(nn.forward(x[0]))
            ConfMat[mapp[y[0]], mapp[hat_y]] += 1
            cr    = (cr*j + (hat_y == y[0]).astype(float))/(j+1)
            # INFO
            sys.stdout.write("\rEvaluation %d/%d" % (j, len(eval_x)))
            sys.stdout.flush()
        print " Accuracy %2.5f %% " % (cr*100),
        print "Polar F-measure average: %0.4f" % Print.posNegFmeasure(ConfMat)
        print ""
    
        print "Testing on Tweets2013"
        # SEMEVAL TRAIN TWEETS
        print "Loading %s" % test_data
        with open(test_data, 'rb') as fid:
            [word2idx_eval, idx2word_eval, eval_x, eval_y] = cPickle.load(fid)
    
        #this not needed anymore (i am doing this conversion when extracting the features)
        # eval_x = [tx.astype('int32') for tx in eval_x]
        eval_y = [np.array(ty).astype('int32')[None] for ty in eval_y]
        
        # Evaluation
        # mapping for the confusion magtrix
        # yo: 0 pos, 1 neg, 2, neu
        #     0 neu, 1 pos, 2, neg
        mapp    = np.array([ 1, 2, 0])
        ConfMat = np.zeros((3, 3))
        cr      = 0.
        for j, x, y in zip(np.arange(len(eval_x)), eval_x, eval_y):
            hat_y = np.argmax(nn.forward(x[0]))
            ConfMat[mapp[y[0]], mapp[hat_y]] += 1
            cr    = (cr*j + (hat_y == y[0]).astype(float))/(j+1)
            # INFO
            sys.stdout.write("\rEvaluation %d/%d" % (j, len(eval_x)))
            sys.stdout.flush()
        print " Accuracy %2.5f %% " % (cr*100),
        print "Polar F-measure average: %0.4f" % Print.posNegFmeasure(ConfMat)
        print ""
