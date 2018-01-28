
# coding: utf-8

# ### Amazon Sentiment Data

# In[ ]:


import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.utils import AmazonData
corpus = srs.SentimentCorpus("books")
data = AmazonData(corpus=corpus)


# ### Implement Pytorch Forward pass

# As the final exercise today implement the log `forward()` method in 
# 
#     lxmls/deep_learning/pytorch_models/mlp.py
# 
# Use the previous exercise as reference. After you have completed this you can run both systems for comparison.

# In[ ]:


# Model
geometry = [corpus.nr_features, 20, 2]
activation_functions = ['sigmoid', 'softmax']

# Optimization
learning_rate = 0.05
num_epochs = 10
batch_size = 30


# In[ ]:


import numpy as np
from lxmls.deep_learning.pytorch_models.mlp import PytorchMLP
model = PytorchMLP(
    geometry=geometry,
    activation_functions=activation_functions,
    learning_rate=learning_rate
)


# In[ ]:


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

