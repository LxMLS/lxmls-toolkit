import codecs
import gzip
from lxmls.sequences.label_dictionary import *
from lxmls.sequences.sequence import *
from lxmls.sequences.sequence_list import *
from os.path import dirname
import numpy as np  # This is also needed for theano=True

# from nltk.corpus import brown

# Directory where the data files are located.
data_dir = dirname(__file__) + "/../../data/"

# Train and test files for english WSJ part of the Penn Tree Bank
en_train = data_dir + "train-02-21.conll"
en_dev = data_dir + "dev-22.conll"
en_test = data_dir + "test-23.conll"

# Train and test files for portuguese Floresta sintatica
pt_train = data_dir + "pt_train.txt"
pt_dev = ""
pt_test = data_dir + "pt_test.txt"


def compacify(train_seq, test_seq, dev_seq, theano=False):
    """
    Create a map for indices that is be compact (do not have unused indices)
    """

    # REDO DICTS
    new_x_dict = LabelDictionary()
    new_y_dict = LabelDictionary(['noun'])
    for corpus_seq in [train_seq, test_seq, dev_seq]:
        for seq in corpus_seq:
            for index in seq.x:
                word = corpus_seq.x_dict.get_label_name(index)
                if word not in new_x_dict:
                    new_x_dict.add(word)
            for index in seq.y:
                tag = corpus_seq.y_dict.get_label_name(index)
                if tag not in new_y_dict:
                    new_y_dict.add(tag)

    # REDO INDICES
    # for corpus_seq in [train_seq2, test_seq2, dev_seq2]:
    for corpus_seq in [train_seq, test_seq, dev_seq]:
        for seq in corpus_seq:
            for i in seq.x:
                if corpus_seq.x_dict.get_label_name(i) not in new_x_dict:
                    pass
            for i in seq.y:
                if corpus_seq.y_dict.get_label_name(i) not in new_y_dict:
                    pass
            seq.x = [new_x_dict[corpus_seq.x_dict.get_label_name(i)] for i in seq.x]
            seq.y = [new_y_dict[corpus_seq.y_dict.get_label_name(i)] for i in seq.y]
            # For compatibility with GPUs store as numpy arrays and cats to int
            # 32
            if theano:
                seq.x = np.array(seq.x, dtype='int32')
                seq.y = np.array(seq.y, dtype='int32')
        # Reinstate new dicts
        corpus_seq.x_dict = new_x_dict
        corpus_seq.y_dict = new_y_dict

        # Add reverse indices
        corpus_seq.word_dict = {v: k for k, v in new_x_dict.items()}
        corpus_seq.tag_dict = {v: k for k, v in new_y_dict.items()}

        # SANITY CHECK:
        # These must be the same
    #    tmap  = {v: k for k, v in train_seq.x_dict.items()}
    #    tmap2 = {v: k for k, v in train_seq2.x_dict.items()}
    #    [tmap[i] for i in train_seq[0].x]
    #    [tmap2[i] for i in train_seq2[0].x]

    return train_seq, test_seq, dev_seq


class PostagCorpus(object):

    def __init__(self):
        # Word dictionary.
        self.word_dict = LabelDictionary()

        # POS tag dictionary.
        # Initialize noun to be tag zero so that it the default tag.
        self.tag_dict = LabelDictionary(['noun'])

        # Initialize sequence list.
        self.sequence_list = SequenceList(self.word_dict, self.tag_dict)

    # Read a text file in conll format and return a sequence list
    #
    def read_sequence_list_conll(self, train_file,
                                 mapping_file=("%s/en-ptb.map"
                                               % dirname(__file__)),
                                 max_sent_len=100000,
                                 max_nr_sent=100000):

        # Build mapping of postags:
        mapping = {}
        if mapping_file is not None:
            for line in open(mapping_file):
                coarse, fine = line.strip().split("\t")
                mapping[coarse.lower()] = fine.lower()
        instance_list = self.read_conll_instances(train_file,
                                                  max_sent_len,
                                                  max_nr_sent, mapping)
        seq_list = SequenceList(self.word_dict, self.tag_dict)
        for sent_x, sent_y in instance_list:
            seq_list.add_sequence(sent_x, sent_y)

        return seq_list

    # ----------
    # Reads a conll file into a sequence list.
    # ----------
    def read_conll_instances(self, file, max_sent_len, max_nr_sent, mapping):
        if file.endswith("gz"):
            zf = gzip.open(file, 'rb')
            reader = codecs.getreader("utf-8")
            contents = reader(zf)
        else:
            contents = codecs.open(file, "r", "utf-8")

        nr_sent = 0
        instances = []
        ex_x = []
        ex_y = []
        nr_types = len(self.word_dict)
        nr_pos = len(self.tag_dict)
        for line in contents:
            toks = line.split()
            if len(toks) < 2:
                # print "sent n %i size %i"%(nr_sent,len(ex_x))
                if len(ex_x) < max_sent_len and len(ex_x) > 1:
                    # print "accept"
                    nr_sent += 1
                    instances.append([ex_x, ex_y])
                # else:
                #     if(len(ex_x) <= 1):
                #         print "refusing sentence of len 1"
                if nr_sent >= max_nr_sent:
                    break
                ex_x = []
                ex_y = []
            else:
                pos = toks[4]
                word = toks[1]
                pos = pos.lower()
                if pos not in mapping:
                    mapping[pos] = "noun"
                    print "unknown tag %s" % pos
                pos = mapping[pos]
                if word not in self.word_dict:
                    self.word_dict.add(word)
                if pos not in self.tag_dict:
                    self.tag_dict.add(pos)
                ex_x.append(word)
                ex_y.append(pos)
                # ex_x.append(self.word_dict[word])
                # ex_y.append(self.tag_dict[pos])
        return instances

    # Read a text file in brown format and return a sequence list
    #
    # def read_sequence_list_brown(self,mapping_file="readers/en-ptb.map",max_sent_len=100000,max_nr_sent=100000,categories=""):
    #     ##Build mapping of postags:
    #     mapping = {}
    #     if(mapping_file != None):
    #         for line in open(mapping_file):
    #             coarse,fine = line.strip().split("\t")
    #             mapping[coarse.lower()] = fine.lower()

    #     if(categories == ""):
    #         sents = brown.tagged_sents()
    #     else:
    #         sents = brown.tagged_sents(categories=categories)
    #     seq_list = Sequence_List(self.word_dict,self.int_to_word,self.tag_dict,self.int_to_tag)
    #     nr_types = len(self.word_dict)
    #     nr_tag = len(self.tag_dict)
    #     for sent in sents:
    #         if(len(sent) > max_sent_len or len(sent) <= 1):
    #             continue
    #         ns_x = []
    #         ns_y = []
    #         for word,tag in sent:
    #                 tag = tag.lower()
    #                 if(tag not in mapping):
    #                     ##Add unk tags to dict
    #                     mapping[tag] = "noun"
    #                 c_t =  mapping[tag]
    #                 if(word not in self.word_dict):
    #                     self.word_dict[word] = nr_types
    #                     c_word = nr_types
    #                     self.int_to_word.append(word)
    #                     nr_types += 1
    #                 else:
    #                     c_word = self.word_dict[word]
    #                 if(c_t not in self.tag_dict):
    #                     self.tag_dict[c_t] = nr_tag
    #                     c_pos_c = nr_tag
    #                     self.int_to_tag.append(c_t)
    #                     nr_tag += 1
    #                 else:
    #                     c_pos_c = self.tag_dict[c_t]
    #                 ns_x.append(c_word)
    #                 ns_y.append(c_pos_c)
    #         seq_list.add_sequence(ns_x,ns_y)
    #     return seq_list

    # Dumps a corpus into a file
    def save_corpus(self, dir):
        if not os.path.isdir(dir + "/"):
            os.mkdir(dir + "/")
        word_fn = codecs.open(dir + "word.dic", "w", "utf-8")
        for word_id, word in enumerate(self.int_to_word):
            word_fn.write("%i\t%s\n" % (word_id, word))
        word_fn.close()
        tag_fn = open(dir + "tag.dic", "w")
        for tag_id, tag in enumerate(self.int_to_tag):
            tag_fn.write("%i\t%s\n" % (tag_id, tag))
        tag_fn.close()
        word_count_fn = open(dir + "word.count", "w")
        for word_id, counts in self.word_counts.iteritems():
            word_count_fn.write("%i\t%s\n" % (word_id, counts))
        word_count_fn.close()
        self.sequence_list.save(dir + "sequence_list")

    # Loads a corpus from a file
    def load_corpus(self, dir):
        word_fn = codecs.open(dir + "word.dic", "r", "utf-8")
        for line in word_fn:
            word_nr, word = line.strip().split("\t")
            self.int_to_word.append(word)
            self.word_dict[word] = int(word_nr)
        word_fn.close()
        tag_fn = open(dir + "tag.dic", "r")
        for line in tag_fn:
            tag_nr, tag = line.strip().split("\t")
            if tag not in self.tag_dict:
                self.int_to_tag.append(tag)
                self.tag_dict[tag] = int(tag_nr)
        tag_fn.close()
        word_count_fn = open(dir + "word.count", "r")
        for line in word_count_fn:
            word_nr, word_count = line.strip().split("\t")
            self.word_counts[int(word_nr)] = int(word_count)
        word_count_fn.close()
        self.sequence_list.load(dir + "sequence_list")
