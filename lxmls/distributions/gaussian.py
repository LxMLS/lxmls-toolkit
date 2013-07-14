import numpy as np

###############
## Class for univariate gaussian
## p(x) = 1/sqrt(2*pi*simga^2) * e ^ - (x-miu)^2/2*sigma^2
## Where miu is the gaussian mean, and sigma^2 is the gaussian variance
################


class Gaussian():

    def __init__(self,mean,variance):
        self.mean = mean;
        self.variance = variance;

    def sample(self,points):
        return np.random.normal(self.mean,self.variance,points)

### Returns the mean and the variance of a data set of X points assuming that the points come from a gaussian distribution
### X
def estimate_gaussian(X):
    mean = np.mean(X,0)
    variance = np.var(X,0)
    return Gaussian(mean,variance)
    
