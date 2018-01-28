
# coding: utf-8

# ### Amazon Sentiment Data

# In[ ]:


import numpy as np
import lxmls.readers.sentiment_reader as srs
from lxmls.deep_learning.utils import AmazonData
corpus = srs.SentimentCorpus("books")
data = AmazonData(corpus=corpus)


# ### Exercise 2.2 Implement Backpropagation for an MLP in Numpy and train it

# In[ ]:


# Model
geometry = [corpus.nr_features, 20, 2]
activation_functions = ['sigmoid', 'softmax']

# Optimization
learning_rate = 0.05
num_epochs = 10
batch_size = 30


# In[ ]:


from lxmls.deep_learning.numpy_models.mlp import NumpyMLP
model = NumpyMLP(
    geometry=geometry,
    activation_functions=activation_functions,
    learning_rate=learning_rate
)


# #### Milestone 1:
# Check gradients using the empirical gradient computation

# Plot the loss values for the study weight and perturbed versions. Take into account that the gradient for the first layer (embeddings) will be zero unless the word is in the input.

# In[ ]:


from lxmls.deep_learning.mlp import get_mlp_parameter_handlers, get_mlp_loss_range

# Get functions to get and set values of a particular weight of the model
get_parameter, set_parameter = get_mlp_parameter_handlers(
    layer_index=1,
    is_bias=False,
    row=0, 
    column=0
)

# Get batch of data
batch = data.batches('train', batch_size=batch_size)[0]

# Get loss and weight value
current_loss = model.cross_entropy_loss(batch['input'], batch['output'])
current_weight = get_parameter(model.parameters)

# Get range of values of the weight and loss around current parameters values
weight_range, loss_range = get_mlp_loss_range(model, get_parameter, set_parameter, batch)


# In[ ]:


# Use this to know the non-zero values of the input (that have non-zero gradient)
batch['input'][0].nonzero()


# In[ ]:


# Get the gradient value for that weight
current_gradient = get_parameter((model.backpropagation(batch['input'], batch['output'])))


# In[ ]:


import matplotlib.pyplot as plt
# Plot empirical
plt.plot(weight_range, loss_range)
plt.plot(current_weight, current_loss, 'xr')
plt.ylabel('loss value')
plt.xlabel('weight value')
# Plot real
h = plt.plot(
    weight_range,
    current_gradient*(weight_range - current_weight) + current_loss, 
    'r--'
)
plt.show()


# #### Milestone 2:
# Train a MLP

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

