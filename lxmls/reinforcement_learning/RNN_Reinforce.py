from IPython import embed
from lxmls.readers.pos_corpus import PostagCorpusData
from lxmls.deep_learning.pytorch_models.rnn import PolicyRNN
import numpy as np
import time

if __name__ == "__main__" :
    # Load Part-of-Speech data
    data = PostagCorpusData()
    print(data.input_size)
    print(data.output_size)
    print(data.maxL)

    model = PolicyRNN(
                  input_size=data.input_size,
                  embedding_size=50,
                  hidden_size=100,
                  output_size=data.output_size,
                  learning_rate=0.05,
                  gamma=0.8,
                  maxL=data.maxL
                  )


    num_epochs = 15
    batch_size = 1
    # Get batch iterators for train and test
    train_batches = data.sample('train', batch_size=batch_size)
    dev_set = data.sample('dev', batch_size=batch_size)
    test_set = data.sample('test', batch_size=batch_size)
    print("RL")

    # Epoch loop
    start = time.time()
    for epoch in range(num_epochs):
        # Batch loop
        for batch in train_batches:
            model.update(input=batch['input'], output=batch['output'])
        # Evaluation dev
        is_hit = []
        for batch in dev_set:
            loss = model.predict_loss(batch['input'], batch['output'])
            is_hit.extend(loss)
        accuracy = 100 * np.mean(is_hit)
        # Inform user
        print("Epoch %d: dev accuracy %2.2f %%" % (epoch + 1, accuracy))

    print("Training took %2.2f seconds per epoch" % ((time.time() - start) / num_epochs))
    # Evaluation test
    is_hit = []
    for batch in test_set:
        is_hit.extend(model.predict_loss(batch['input'],batch['output']))
    accuracy = 100 * np.mean(is_hit)

    # Inform user
    print("Test accuracy %2.2f %%" % accuracy)
