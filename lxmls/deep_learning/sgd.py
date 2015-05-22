import sys
import numpy as np
import time

def class_acc(hat_y, y_ref):
    '''
    Computes percent accuracy and log probability given estimated and reference 
    class indices 
    '''
    # Check probability of devel set 
    pred     = hat_y[y_ref, np.arange(y_ref.shape[0])]
    p_dev    = np.sum(np.log(pred))
    # Check percent correct classification on the devel set
    cr       = np.sum((np.argmax(hat_y, 0) == y_ref).astype(int))*1.0/y_ref.shape[0]
    return (cr, p_dev)


def SGD_train(mlp, devel_set=None, train_set=None, n_iter=20, batch_size=None, 
              lrate=0.01):
    '''
    Stochastic Gradient Descent training
    '''

    # In the manual mode we need to define train set, batch size and
    if getattr(mlp, "train_batch", None):
        n_batch = mlp.n_batch
    else:
        if (not train_set) or (not batch_size):
            raise ValueError, ("In manual mode you need to define train_set"
                               "and batch_size") 
        # Get sizes and check for coherence 
        L       = train_set[0].shape[1]   
        n_batch = L/batch_size         
        if L < batch_size:
            raise ValueError, ("Batch size %d too large for %d train examples"
                               % (batch_size, L)) 

    # For each iteration run backpropagation in a batch of examples. For
    # each batch, sum up all gradients and update each weights with the
    # SGD rule.
    prev_p_devel = None
    prev_p_train = None
    for i in np.arange(n_iter): 
        # This will hold the posterior of train data for each epoch
        p_train   = 0
        init_time = time.clock()
        for n in np.arange(n_batch): 

             # Compiled batch update          
             if getattr(mlp, "train_batch", None):
                 # This updates the model parameters as well!!
                 p_train += -mlp.train_batch(n)

             # Manual batch update          
             else:
                 # Get an index to elements of this batch
                 idx = np.arange(n*batch_size, np.minimum((n+1)*batch_size, L)) 
                 # Get gradients for each layer and this batch
                 delta_weights = mlp.backprop_grad(train_set[0][:, idx], 
                                                   train_set[1][idx])
                 # Update with SGD rule
                 for m in np.arange(mlp.n_layers):
                     # Watch out for sparse matrix
                     mlp.weights[m][0] -= lrate*delta_weights[m][0]
                     # Bias
                     mlp.weights[m][1] -= lrate*delta_weights[m][1]

             # INFO
             sys.stdout.write("\rBatch %d/%d (%d%%) " % (n+1, n_batch, (n+1)*100.0/n_batch))
             sys.stdout.flush()
        batch_time = time.clock() - init_time
        # Check probability of devel set 
        if devel_set:
            corr, p_devel = class_acc(mlp.forward(devel_set[0]), devel_set[1])
            if prev_p_devel:
                delta_p_devel = p_devel - prev_p_devel 
            else:
                delta_p_devel = 0 
            prev_p_devel  = p_devel
        if prev_p_train:
            delta_p_train = p_train - prev_p_train
        else:
            delta_p_train = 0
        prev_p_train    = p_train
        validation_time = time.clock() - init_time - batch_time
        sys.stdout.write("  Epoch %2d/%2d in %2.2f seg\n" % (i+1, n_iter, batch_time))
        if devel_set:
            sys.stdout.write("Logpos devel: %10.1f (delta: %10.2f) Corr devel %2.2f\n\n" % (p_devel, delta_p_devel, corr))
    print ""          
