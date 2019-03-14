from IPython import embed

# Load Part-of-Speech data
from lxmls.readers.pos_corpus import PostagCorpusData
data = PostagCorpusData()

print(data.input_size)
print(data.output_size)
print(data.maxL)


# Alterbative native CuDNN native implementation of RNNs
from lxmls.deep_learning.pytorch_models.rnn import PolicyRNN
model = PolicyRNN(
    input_size=data.input_size,
    embedding_size=50,
    hidden_size=20,
    output_size=data.output_size,
    learning_rate=0.1,
    gamma=0.9,
    RL=True,
    maxL=data.maxL
)

num_epochs = 10

import numpy as np
import time


# Get batch iterators for train and test
train_batches = data.sample('train', batch_size=10)
dev_set = data.sample('dev', batch_size=10)
test_set = data.sample('test', batch_size=10)

# # Epoch loop
# start = time.time()
# for epoch in range(num_epochs):
#
#     # Batch loop
#     for batch in train_batches:
#         model.update(input=batch['input'], output=batch['output'])
#
#     # Evaluation dev
#     is_hit = []
#     for batch in dev_set:
#         is_hit.extend(model.predict(input=batch['input']) == batch['output'])
#     accuracy = 100 * np.mean(is_hit)
#
#     # Inform user
#     print("Epoch %d: dev accuracy %2.2f %%" % (epoch + 1, accuracy))
#
# print("Training took %2.2f seconds per epoch" % ((time.time() - start) / num_epochs))
#
# # Evaluation test
# is_hit = []
# for batch in test_set:
#     is_hit.extend(model.predict(input=batch['input']) == batch['output'])
# accuracy = 100 * np.mean(is_hit)
#
# # Inform user
# print("Test accuracy %2.2f %%" % accuracy)
#
# # Example of sampling
# print(train_batches[3]['input'])
# samples, log_probs = model._sample(input=train_batches[3]['input'])
# samples, log_probs
print("RL")
# Epoch loop
start = time.time()
for epoch in range(num_epochs):
    # Batch loop
    for batch in train_batches:
        #TODO: Use this here to create an RL inside model.update()
        #samples, log_probs = model._sample(input=batch['input'])  # sample actions and its neg log probs
        embed()
        model.update(input=batch['input'], output=batch['output'])

    # Evaluation dev
    is_hit = []
    for batch in dev_set:
        is_hit.extend(model.predict(input=batch['input']) == batch['output'])
    accuracy = 100 * np.mean(is_hit)

    # Inform user
    print("Epoch %d: dev accuracy %2.2f %%" % (epoch + 1, accuracy))

print("Training took %2.2f seconds per epoch" % ((time.time() - start) / num_epochs))

# Evaluation test
is_hit = []
for batch in test_set:
    is_hit.extend(model.predict(input=batch['input']) == batch['output'])
accuracy = 100 * np.mean(is_hit)

# Inform user
print("Test accuracy %2.2f %%" % accuracy)
