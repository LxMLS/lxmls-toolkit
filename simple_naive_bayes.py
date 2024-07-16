from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import Counter, defaultdict
from math import log

dataset = load_dataset("emad12/stock_tweets_sentiment", split="train")
indices = list(range(len(dataset)))
shuffle(indices)
train_idx = indices[:-2000]
dev_idx = indices[-2000:-1000]
test_idx = indices[-1000:]

# output domain
sentiments = [-1, 0, 1]

smooth = 1

# train
prior_counts = Counter()
conditional_counts = defaultdict(lambda: Counter())
for i in tqdm(train_idx, desc='training'):
    # strong independence assumption each word affects sentiment independent of
    # others
    tokens = dataset[i]['tweet'].split()
    # update counts
    conditional_counts[dataset[i]['sentiment']].update(tokens)
    prior_counts.update([dataset[i]['sentiment']])

# inference
is_correct = []
for i in tqdm(dev_idx, desc='inference'):
    # strong independence assumption each word affects sentiment independent of
    # others
    tokens = dataset[i]['tweet'].split()

    # compute joint probability for all classes and tweet
    scores = []
    for sentiment in sentiments:
        normalizer = sum(conditional_counts[sentiment].values())
        size = len(conditional_counts[sentiment].values())
        log_joint = log(prior_counts[sentiment])
        for w in tokens:
            log_joint += log(conditional_counts[sentiment][w] + smooth)
            log_joint -= log(normalizer + smooth * size)
        scores.append(log_joint)

    # pick the highest probability class
    best_index = np.argmax(scores)
    # store if we chose right or wrong
    is_correct.append(sentiments[best_index] == dataset[i]['sentiment'])

print('{:.3f}'.format(np.mean(is_correct)))
