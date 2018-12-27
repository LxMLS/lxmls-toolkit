
# coding: utf-8

# ### Amazon Sentiment Data

# In[ ]:


import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.utils import AmazonData
corpus = srs.SentimentCorpus("books")
data = AmazonData(corpus=corpus)


# ### Train Log Linear in Pytorch
# In order to learn the differences between a numpy and a Pytorch implementation, explore the reimplementation of Ex. 3.1 in Pytorch. Compare the content of each of the functions, in particular the `forward()` and `update()` methods. The comments indicated as IMPORTANT will highlight common sources of errors.

# In[ ]:


from lxmls.deep_learning.utils import Model, glorot_weight_init
import numpy as np
import torch
from torch.autograd import Variable

class PytorchLogLinear(Model):
    
    def __init__(self, **config):
        
        # Initialize parameters
        weight_shape = (config['input_size'], config['num_classes'])
        # after Xavier Glorot et al
        self.weight = glorot_weight_init(weight_shape, 'softmax')
        self.bias = np.zeros((1, config['num_classes']))
        self.learning_rate = config['learning_rate']
        
        # IMPORTANT: Cast to pytorch format
        self.weight = Variable(torch.from_numpy(self.weight).float(), requires_grad=True)
        self.bias = Variable(torch.from_numpy(self.bias).float(), requires_grad=True)
        
        # Instantiate softmax and negative logkelihood in log domain
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss()
        
    def _log_forward(self, input=None):  
        """Forward pass of the computation graph in logarithm domain (pytorch)"""
        
        # IMPORTANT: Cast to pytorch format
        input = Variable(torch.from_numpy(input).float(), requires_grad=False)
        
        # Linear transformation
        z =  torch.matmul(input, torch.t(self.weight)) + self.bias
        
        # Softmax implemented in log domain
        log_tilde_z = self.logsoftmax(z)
        
        # NOTE that this is a pytorch class!
        return log_tilde_z
            
    def predict(self, input=None):
        """Most probably class index"""
        log_forward = self._log_forward(input).data.numpy()
        return np.argmax(np.exp(log_forward), axis=1)
        
    def update(self, input=None, output=None):
        """Stochastic Gradient Descent update"""
        
        # IMPORTANT: Class indices need to be casted to LONG
        true_class = Variable(torch.from_numpy(output).long(), requires_grad=False)
        
        # Compute negative log-likelihood loss
        loss = self.loss(self._log_forward(input), true_class)
        # Use autograd to compute the backward pass.
        loss.backward()
        
        # SGD update
        self.weight.data -= self.learning_rate * self.weight.grad.data
        self.bias.data -= self.learning_rate * self.bias.grad.data
        
        # Zero gradients
        self.weight.grad.data.zero_()
        self.bias.grad.data.zero_()
        
        return loss.data.numpy()


# Once you understand the model you can instantiate it and run it using the standard training loop we have used on previous exercises.

# In[ ]:


model = PytorchLogLinear(
    input_size=corpus.nr_features,
    num_classes=2, 
    learning_rate=0.05
)


# In[ ]:


# Hyper-parameters
num_epochs = 10
batch_size = 30

# Get batch iterators for train and test
train_batches = data.batches('train', batch_size=batch_size)
test_set = data.batches('test', batch_size=None)[0]

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

