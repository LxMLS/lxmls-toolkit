import numpy as np

############################################################################
# Functions to compute in log-domain.
############################################################################

def logzero():
    return -1e20
#    return -np.inf

def safe_log(x):
    y       = np.zeros(x.shape)	+ logzero()
    y[x>0]  = np.log(x[x>0]) 
    return y
    
def logsum_pair(logx, logy):
    '''
    Return log(x+y), avoiding arithmetic underflow/overflow.
 
    logx: log(x)
    logy: log(y)
 
    Rationale:
 
    x + y    = e^logx + e^logy
             = e^logx (1 + e^(logy-logx))
    log(x+y) = logx + log(1 + e^(logy-logx)) (1)
 
    Likewise, 
    log(x+y) = logy + log(1 + e^(logx-logy)) (2)
 
    The computation of the exponential overflows earlier and is less precise
    for big values than for small values. Due to the presence of logy-logx
    (resp. logx-logy), (1) is preferred when logx > logy and (2) is preferred
    otherwise.
    '''
    if logx == logzero():
        return logy
    elif logx > logy:
        return logx + np.log1p(np.exp(logy-logx))
#        return logx + np.log(1 + np.exp(logy-logx))
    else:
        return logy + np.log1p(np.exp(logx-logy))
#        return logy + np.log(1 + np.exp(logx-logy))
        
        
#def logsum2(logv):
#    '''
#    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
#    '''
#    res = logzero()
#    for val in logv:
#        res = logsum_pair(res, val)
#    return res

def logsum(logv):
    '''
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    '''
    c = np.max(logv)
    return c + np.log(np.sum(np.exp(logv - c)))
#    res2 = logsum2(logv)
#    import pdb
#    assert (res-res2)**2 < 1e-6, pdb.set_trace()
#    return res
############################################################################
