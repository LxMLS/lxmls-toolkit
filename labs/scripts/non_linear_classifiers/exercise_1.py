
# coding: utf-8

# ### Amazon Sentiment Data

# To ease-up the upcoming implementation exercise, examine and comment the following implementation of a log-linear model and its gradient update rule. Start by loading Amazon sentiment corpus used in day 1

# In[ ]:


import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.utils import AmazonData
corpus = srs.SentimentCorpus("books")
data = AmazonData(corpus=corpus)


# In[ ]:


data.datasets['train']


# ### A Shallow Model: Log-Linear in Numpy

# Compare the following numpy implementation of a log-linear model with the derivations seen in the previous sections. Introduce comments on the blocks marked with # relating them to the corresponding algorithm steps.

# In[ ]:


from lxmls.deep_learning.utils import Model, glorot_weight_init, index2onehot
import numpy as np
from scipy.special import logsumexp

class NumpyLogLinear(Model):
    
    def __init__(self, **config):
        
        # Initialize parameters
        weight_shape = (config['input_size'], config['num_classes'])
        # after Xavier Glorot et al
        self.weight = glorot_weight_init(weight_shape, 'softmax')
        self.bias = np.zeros((1, config['num_classes']))
        self.learning_rate = config['learning_rate']
        
    def log_forward(self, input=None):  
        """Forward pass of the computation graph"""
        
        # Linear transformation
        z = np.dot(input, self.weight.T) + self.bias
        
        # Softmax implemented in log domain
        log_tilde_z = z - logsumexp(z, axis=1)[:, None]
        
        return log_tilde_z
        
    def predict(self, input=None):
        """Prediction: most probable class index"""
        return np.argmax(np.exp(self.log_forward(input)), axis=1)      
     
    def update(self, input=None, output=None):
        """Stochastic Gradient Descent update"""
        
        # Probabilities of each class
        class_probabilities = np.exp(self.log_forward(input))
        batch_size, num_classes = class_probabilities.shape
        
        # Error derivative at softmax layer
        I = index2onehot(output, num_classes)
        error = (class_probabilities - I) / batch_size
        
        # Weight gradient
        gradient_weight = np.zeros(self.weight.shape)
        for l in range(batch_size):
            gradient_weight += np.outer(error[l, :], input[l, :])
        
        # Bias gradient
        gradient_bias = np.sum(error, axis=0, keepdims=True)
        
        # SGD update
        self.weight = self.weight - self.learning_rate * gradient_weight
        self.bias = self.bias - self.learning_rate * gradient_bias


# ### Training Bench

# Instantiate model and data classes. Check the initial accuracy of the model. This should be close to 50% since we are on a binary prediction task and the model is not trained yet.

# In[ ]:


learning_rate = 0.05
num_epochs = 10
batch_size = 30


# In[ ]:


model = NumpyLogLinear(
    input_size=corpus.nr_features,
    num_classes=2, 
    learning_rate=learning_rate
)


# In[ ]:


# Define number of epochs and batch size
num_epochs = 10
batch_size = 30

# Get batch iterators for train and test
train_batches = data.batches('train', batch_size=batch_size)
test_set = data.batches('test', batch_size=None)[0]

# Get intial accuracy
hat_y = model.predict(input=test_set['input'])
accuracy = 100*np.mean(hat_y == test_set['output'])
print("Initial accuracy %2.2f %%" % accuracy)


# Train the model with simple batch stochastic gradient descent. Be sure to understand each of the steps involved, including the code running inside of the model class. We will be wokring on a more complex version of the model in the upcoming exercise.

# In[ ]:


# Epoch loop
for epoch in range(num_epochs):

    # Batch loop
    for batch in train_batches:
        model.update(input=batch['input'], output=batch['output'])

    # Prediction for this epoch
    hat_y = model.predict(input=test_set['input'])

    # Evaluation
    accuracy = 100*np.mean(hat_y == test_set['output'])

    # Inform user
    print("Epoch %d: accuracy %2.2f %%" % (epoch+1, accuracy))

