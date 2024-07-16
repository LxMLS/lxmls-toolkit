# http://www.scipy.org/SciPyPackages/Sparse
from __future__ import division

import codecs

import numpy as np
from os import path
from collections import OrderedDict


class SentimentCorpus:

    def __init__(self, domain, train_per=0.8, dev_per=0, test_per=0.2):
        X, y, feat_dict, feat_counts = build_dicts(domain)
        self.nr_instances = y.shape[0]
        self.nr_features = X.shape[1]
        self.X = X
        self.y = y
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts
        train_y, dev_y, test_y, train_X, dev_X, test_X = split_train_dev_test(self.X, self.y, train_per, dev_per, test_per)
        self.train_X = train_X
        self.train_y = train_y
        self.dev_X = dev_X
        self.dev_y = dev_y
        self.test_X = test_X
        self.test_y = test_y


def split_train_dev_test(X, y, train_per, dev_per, test_per):
    if train_per+dev_per+test_per > 1:
        print("Train Dev Test split should sum to one")
        return
    dim = y.shape[0]
    split1 = int(dim * train_per)
    if dev_per == 0:
        train_y, test_y = np.vsplit(y, [split1])
        dev_y = np.array([])
        train_X = X[0:split1, :]
        dev_X = np.array([])
        test_X = X[split1:, :]

    else:
        split2 = int(dim * (train_per+dev_per))
        train_y, dev_y, test_y = np.vsplit(y, (split1, split2))
        train_X = X[0:split1, :]
        dev_X = X[split1:split2, :]
        test_X = X[split2:, :]
    return train_y, dev_y, test_y, train_X, dev_X, test_X


_base_sentiment_dir = path.join(path.dirname(__file__), "..", "..", "data", "sentiment")


def build_dicts(domain):
    """Builds feature dictionaries for a given domain of the sentiment analysis corpus."""
    sentiment_domains = ["books", "dvd", "electronics", "kitchen"]
    feat_counts = OrderedDict()
    if domain not in sentiment_domains:
        print(
            "Domain does not exist: \"%s\": Available are: %s" % 
            (domain, sentiment_domains)
        )
        return

    # Build Dictionarie wit counts
    nr_pos = 0
    pos_file = codecs.open(path.join(_base_sentiment_dir, domain, "positive.review"), 'r', 'utf8')
    for line in pos_file:
        nr_pos += 1
        toks = line.split(" ")
        for feat in toks[0:-1]:
            name, counts = feat.split(":")
            if name not in feat_counts:
                feat_counts[name] = 0
            feat_counts[name] += int(counts)
    pos_file.close()
    nr_neg = 0
    neg_file = codecs.open(path.join(_base_sentiment_dir, domain, "negative.review"), 'r', 'utf8')
    for line in neg_file:
        nr_neg += 1
        toks = line.split(" ")
        for feat in toks[0:-1]:
            name, counts = feat.split(":")
            if name not in feat_counts:
                feat_counts[name] = 0
            feat_counts[name] += int(counts)
    neg_file.close()

    # Build X,y data
    # To read is better in linked list format (lil)
    size = nr_pos + nr_neg
    # print "Before removing %i %i"%(len(feat_counts),sum(feat_counts.values()))
    # Remove all features that occur less than X
    to_remove = []
    for key, value in feat_counts.items():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]

    nr_feat = len(feat_counts)
    # print nr_feat
    feat_dict = OrderedDict()
    i = 0
    # print "After removing %i %i"%(len(feat_counts),sum(feat_counts.values()))
    for key in list(feat_counts.keys()):
        feat_dict[key] = i
        i += 1

    X = np.zeros((size, nr_feat), dtype=float)
    y = np.vstack((np.zeros([nr_pos, 1], dtype=int), np.ones([nr_neg, 1], dtype=int)))
    pos_file = codecs.open(path.join(_base_sentiment_dir, domain, "positive.review"), 'r', 'utf8')
    nr_pos = 0
    for line in pos_file:
        toks = line.split(" ")
        for feat in toks[0:-1]:
            name, counts = feat.split(":")
            if name in feat_dict:
                # print "adding %s with counts %s"%(name,counts)
                X[nr_pos, feat_dict[name]] = int(counts)
        nr_pos += 1
    neg_file = codecs.open(path.join(_base_sentiment_dir, domain, "negative.review"), 'r', 'utf8')
    nr_neg = 0
    for line in neg_file:
        toks = line.split(" ")
        for feat in toks[0:-1]:
            name, counts = feat.split(":")
            if name in feat_dict:
                # print "adding %s with counts %s"%(name,counts)
                X[nr_pos+nr_neg, feat_dict[name]] = int(counts)
        nr_neg += 1
    # print X.shape
    # print np.sum(X)
    X_aux = X.copy()
    y_aux = y.copy()
    # Mix positive and negative examples
    half_instances = (nr_pos+nr_neg) // 2
    positive_index = np.arange(half_instances)

    X[2*positive_index] = X_aux[positive_index]
    y[2*positive_index] = y_aux[positive_index]
    X[2*positive_index+1] = X_aux[positive_index+half_instances]
    y[2*positive_index+1] = y_aux[positive_index+half_instances]
    return X, y, feat_dict, feat_counts
