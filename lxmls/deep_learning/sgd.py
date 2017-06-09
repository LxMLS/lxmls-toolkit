from __future__ import division
import sys
import numpy as np
import time
import theano
import theano.tensor as T


def get_numpy_batch_update(model, lrate, theano=False):
    '''
    returns a batch update function with the update rule written in numpy 
    (slow)
    '''

    # Note that the model and lrates variables will be available inside
    # batch_up. This happend because a function has acees to the variables in
    # the scope it was defined (where this comment is) This is known as a
    # closure.

    if theano:

        def batch_up(batch_x, batch_y):
            '''
            Update for theano.shared variables
            '''
            # Get gradients for each layer and this batch
            nabla_params = model.grads(batch_x, batch_y)
            # Update each parameter with SGD rule
            for m in np.arange(len(model.params)):
                # Parameters as theano shared variables
                model.params[m].set_value(model.params[m].get_value() 
                                          - lrate*np.array(nabla_params[m]))

    else:

        def batch_up(batch_x, batch_y):
            '''
            Update for numpy arrays 
            '''
            # Get gradients for each layer and this batch
            # Get gradients for each layer and this batch
            nabla_params = model.grads(batch_x, batch_y)
            # Update each parameter with SGD rule
            for m in np.arange(len(model.params)):
                # Parameters as theano shared variables
                model.params[m] -= lrate * nabla_params[m]

    return batch_up


def class_acc(hat_y, y_ref):
    """
    Computes percent accuracy and log probability given estimated and reference
    class indices
    """
    # Check probability of devel set
    pred = hat_y[y_ref, np.arange(y_ref.shape[0])]
    p_dev = np.sum(np.log(pred))
    # Check percent correct classification on the devel set
    cr = np.sum((np.argmax(hat_y, 0) == y_ref).astype(int)) / y_ref.shape[0]
    return cr, p_dev


def sanity_checks(batch_up, n_batch, bsize, lrate, train_set):

    if batch_up:

        if not n_batch or not bsize or not train_set:
            raise ValueError("If you use compiled batch update you need to "
                             "specify n_batch, bsize and train_set")
    else:

        if not bsize or not lrate or not train_set:
            raise ValueError("If compiled batch not used you need to specity"
                             "bsize, lrate and train_set")


def SGD_train(model, n_iter, bsize=None, lrate=None, train_set=None,
              batch_up=None, n_batch=None, devel_set=None, model_dbg=None):

    # SANITY CHECKS:
    sanity_checks(batch_up, n_batch, bsize, lrate, train_set)

    train_x, train_y = train_set

    # If no batch update provided we use the numpy one
    if not batch_up:
        if getattr(model, "_forward", None):
            batch_up = get_numpy_batch_update(model, lrate, theano=True)
        else:    
            batch_up = get_numpy_batch_update(model, lrate, theano=False)
        
    # Number of mini batches
    n_batch = int(np.ceil(float(train_x.shape[1])/bsize))

    # For each iteration run backpropagation in a batch of examples. For
    # each batch, sum up all gradients and update each weights with the
    # SGD rule.
    prev_p_devel = None
    prev_p_train = None
    for i in np.arange(n_iter):
        # This will hold the posterior of train data for each epoch
        init_time = time.clock()
        for j in np.arange(n_batch):
            # Mini batch
            batch_x = train_x[:, j*bsize:(j+1)*bsize]
            batch_y = train_y[j*bsize:(j+1)*bsize]
            # Update parameters
            batch_up(batch_x, batch_y)
            # INFO
            print "\rBatch %d/%d (%d%%) " % \
                  (j+1, n_batch, (j+1)*100.0/n_batch),
            sys.stdout.flush()

        # Record time
        batch_time = time.clock() - init_time

        # Check probability of devel set
        if devel_set:
            corr, p_devel = class_acc(model.forward(devel_set[0]), devel_set[1])
            if prev_p_devel:
                delta_p_devel = p_devel - prev_p_devel
            else:
                delta_p_devel = 0
            prev_p_devel = p_devel

        validation_time = time.clock() - init_time - batch_time
        sys.stdout.write("  Epoch %2d/%2d in %2.2f seg\n" % (i+1, n_iter, batch_time))
        if devel_set:
            sys.stdout.write(
                "Logpos devel: %10.1f (delta: %10.2f) Corr devel %2.2f\n\n" %
                (p_devel, delta_p_devel, corr))
    print ""
