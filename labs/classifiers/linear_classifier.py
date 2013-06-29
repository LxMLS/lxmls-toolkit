import numpy as np

class LinearClassifier():

    def __init__(self):
        self.trained = False

    def train(self,x,y):
        '''Returns the weight vector.

        '''
        raise NotImplementedError('LinearClassifier.train not implemented')

    def get_scores(self,x,w):
        '''Computes the dot product between x and w.

        '''
        return np.dot(x,w)

    def get_label(self,x,w):
        ''' Computes the label for each data point.

        '''
        scores = np.dot(x,w)
        return np.argmax(scores,axis=1).transpose()

    def test(self,x,w):
        '''Classifies the points 'x' based on a weight vector 'w'.

        '''
        if self.trained == False:
            raise ValueError("Model not trained. Cannot test")
            return 0
        x = self.add_intercept_term(x)
        return self.get_label(x,w)
    
    def add_intercept_term(self,x):
        '''Adds a column of ones to estimate the intercept term for separation
        boundary.

        '''
        nr_x,nr_f = x.shape
        intercept = np.ones([nr_x,1])
        x = np.hstack((intercept,x))
        return x

    def evaluate(self,truth,predicted):
        '''Computes the accuracy, given the 'predicted' and the 'thruth'.

        '''
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total
