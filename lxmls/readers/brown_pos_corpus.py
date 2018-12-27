from nltk.corpus import brown
import sys
from sequences.sequence import *
from sequences.sequence_list import *


class BrownPostag:

    def __init__(
            self,
            max_sent_len=15,
            train_sents=1000,
            dev_sents=200,
            test_sents=200,
            mapping_file="readers/en-brown.map"):

        # Build mapping of postags:
        mapping = {}
        if mapping_file is not None:
            for line in open(mapping_file):
                coarse, fine = line.strip().split("\t")
                mapping[coarse.lower()] = fine.lower()
        x_dict = {}
        int_to_word = []
        y_dict = {}
        int_to_pos = []

        max_sents = train_sents+dev_sents+test_sents
        sents = brown.tagged_sents()
        train_s = sents[0:train_sents]
        dev_s = sents[train_sents:train_sents+dev_sents]
        test_s = sents[train_sents+dev_sents:train_sents+dev_sents+test_sents]
        word_c = 0
        tag_c = 0
        # Initialize noun to be tag zero so that it the default tag
        y_dict["noun"] = 0
        int_to_pos = ["noun"]
        tag_c += 1
        train_s_x = []
        train_s_y = []
        dev_s_x = []
        dev_s_y = []
        test_s_x = []
        test_s_y = []
        word_counts = {}
        seq_list_train = Sequence_List(x_dict, int_to_word, y_dict, int_to_pos)
        seq_list_dev = Sequence_List(x_dict, int_to_word, y_dict, int_to_pos)
        seq_list_test = Sequence_List(x_dict, int_to_word, y_dict, int_to_pos)
        for ds, sl in [[train_s, seq_list_train], [dev_s, seq_list_dev], [test_s, seq_list_test]]:
            for s in ds:
                if len(s) > max_sent_len or len(s) <= 1:
                    continue
                ns_x = []
                ns_y = []
                for word, tag in s:
                    tag = tag.lower()
                    if tag not in mapping:
                        # Add unk tags to dict
                        mapping[tag] = "noun"
                    c_t = mapping[tag]
                    if word not in x_dict:
                        x_dict[word] = word_c
                        c_word = word_c
                        word_counts[c_word] = 1
                        int_to_word.append(word)
                        word_c += 1
                    else:
                        c_word = x_dict[word]
                        word_counts[c_word] += 1
                    if c_t not in y_dict:
                        y_dict[c_t] = tag_c
                        c_pos_c = tag_c
                        int_to_pos.append(c_t)
                        tag_c += 1
                    else:
                        c_pos_c = y_dict[c_t]
                    ns_x.append(c_word)
                    ns_y.append(c_pos_c)

                sl.add_sequence(ns_x, ns_y)
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.int_to_word = int_to_word
        self.int_to_pos = int_to_pos
        self.train = seq_list_train
        self.dev = seq_list_dev
        self.test = seq_list_test
        self.feature_extracted = True
        self.word_counts = word_counts
