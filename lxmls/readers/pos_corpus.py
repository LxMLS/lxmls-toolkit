import codecs
import gzip
from itertools import chain
from random import shuffle

from lxmls.sequences.label_dictionary import *
from lxmls.sequences.sequence import *
from lxmls.sequences.sequence_list import *
from lxmls import data
from os.path import dirname
import numpy as np  # This is also needed for theano=True

# from nltk.corpus import brown

# Train and test files for english WSJ part of the Penn Tree Bank
data.find('train-02-21.conll')
data.find('dev-22.conll')
data.find('test-23.conll')

# Train and test files for portuguese Floresta sintatica
data.find('pt_train.txt')
pt_dev = ""
data.find('pt_test.txt')


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
            seq.x = [
                new_x_dict[corpus_seq.x_dict.get_label_name(i)] for i in seq.x
            ]
            seq.y = [
                new_y_dict[corpus_seq.y_dict.get_label_name(i)] for i in seq.y
            ]
            # For compatibility with GPUs store as numpy arrays and cats to int
            # 32
            if theano:
                seq.x = np.array(seq.x, dtype='int32')
                seq.y = np.array(seq.y, dtype='int32')
        # Reinstate new dicts
        corpus_seq.x_dict = new_x_dict
        corpus_seq.y_dict = new_y_dict

        # Add reverse indices
        corpus_seq.word_dict = {v: k for k, v in list(new_x_dict.items())}
        corpus_seq.tag_dict = {v: k for k, v in list(new_y_dict.items())}

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

    def read_sequence_list_conll(self, train_file,
                                 mapping_file=(
                                    "%s/en-ptb.map" % dirname(__file__)
                                 ),
                                 max_sent_len=100000,
                                 max_nr_sent=100000):
        """Read a text file in conll format and return a sequence list"""

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

    def read_conll_instances(self, file, max_sent_len, max_nr_sent, mapping):
        """Reads a conll file into a sequence list."""
        if file.endswith("gz"):
            zf = gzip.open(file, 'rb')
            reader = codecs.getreader("utf-8")
            contents = reader(zf)
        else:
            contents = open(file, "r", "utf-8")

        nr_sent = 0
        instances = []
        ex_x = []
        ex_y = []
        nr_types = len(self.word_dict)
        nr_pos = len(self.tag_dict)
        for line in contents:
            toks = line.split()
            if len(toks) < 2:
                if len(ex_x) < max_sent_len and len(ex_x) > 1:
                    nr_sent += 1
                    instances.append([ex_x, ex_y])
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
                    print("unknown tag %s" % pos)
                pos = mapping[pos]
                if word not in self.word_dict:
                    self.word_dict.add(word)
                if pos not in self.tag_dict:
                    self.tag_dict.add(pos)
                ex_x.append(word)
                ex_y.append(pos)
        return instances

    def save_corpus(self, dir):
        """Dumps a corpus into a file"""
        if not os.path.isdir(dir + "/"):
            os.mkdir(dir + "/")
        word_fn = open(dir + "word.dic", "w", "utf-8")
        for word_id, word in enumerate(self.int_to_word):
            word_fn.write("%i\t%s\n" % (word_id, word))
        word_fn.close()
        tag_fn = open(dir + "tag.dic", "w")
        for tag_id, tag in enumerate(self.int_to_tag):
            tag_fn.write("%i\t%s\n" % (tag_id, tag))
        tag_fn.close()
        word_count_fn = open(dir + "word.count", "w")
        for word_id, counts in self.word_counts.items():
            word_count_fn.write("%i\t%s\n" % (word_id, counts))
        word_count_fn.close()
        self.sequence_list.save(dir + "sequence_list")

    def load_corpus(self, dir):
        """Loads a corpus from a file"""
        word_fn = open(dir + "word.dic", "r", "utf-8")
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

class PostagCorpusData():
    """WSJ Wrapper using Data() conventions"""

    def __init__(self, **config):

        corpus = PostagCorpus()
        train_seq = corpus.read_sequence_list_conll(data.find('train-02-21.conll'), max_sent_len=15, max_nr_sent=1000)
        dev_seq = corpus.read_sequence_list_conll(data.find('dev-22.conll'), max_sent_len=15, max_nr_sent=1000)
        test_seq = corpus.read_sequence_list_conll(data.find('test-23.conll'), max_sent_len=15, max_nr_sent=1000)

        # Redo indices so that they are consecutive. Also cast all data to numpy arrays
        # of int32 for compatibility with GPUs and theano and add reverse index
        train_seq, test_seq, dev_seq = compacify(train_seq, test_seq, dev_seq, theano=True)

        # Get number of words and tags in the corpus
        self.input_size = len(train_seq.x_dict)
        self.output_size = len(train_seq.y_dict)

        # Data-sets
        self.datasets = {
            'train': {
                'input': [np.array(seq.x) for seq in train_seq],
                'output': [np.array(seq.y) for seq in train_seq]
            },
            'dev': {
                'input': [np.array(seq.x) for seq in dev_seq],
                'output': [np.array(seq.y) for seq in dev_seq]
            },
            'test': {
                'input': [np.array(seq.x) for seq in test_seq],
                'output': [np.array(seq.y) for seq in test_seq]
            }
        }
        # Config
        self.config = config
        # Number of samples
        self.nr_samples = {
           sset: len(content['output'])
           for sset, content in self.datasets.items()
        }
        self.maxL = max(chain(*[[len(seq) for seq in content['input']] for content in self.datasets.values()]))
        return

    def size(self, set_name):
        return self.nr_samples[set_name]

    def batches(self, set_name, batch_size=None):

        assert batch_size == 1, "Only batch_size 1 supported"

        dset = self.datasets[set_name]
        nr_examples = self.nr_samples[set_name]
        if batch_size is None:
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples*1./batch_size))

        data = []
        for batch_n in range(nr_batch):
            # Colect data for this batch
            data_batch = {}
            for side in ['input', 'output']:
                data_batch[side] = np.array(dset[side][
                   batch_n * batch_size:(batch_n + 1) * batch_size
                ])[0, :]
            data.append(data_batch)

        return DataIterator(data, nr_samples=self.nr_samples[set_name])


    def sample(self, set_name, batch_size=None):
        dset = self.datasets[set_name]
        nr_examples = self.nr_samples[set_name]
        if batch_size is None:
            nr_batch = 1
            batch_size = nr_examples
        else:
            nr_batch = int(np.ceil(nr_examples*1./batch_size))
        data = []
        for batch_n in range(nr_batch):
            #Colect data for this batch
            data_batch = {}
            sample = np.random.randint(0, nr_examples, batch_size)
            for side in ['input', 'output']:
                data_batch[side] = np.asarray(dset[side])[sample]
            data.append(data_batch)
        return DataIterator(data, nr_samples=self.nr_samples[set_name])


class DataIterator(object):
    """
    Basic data iterator
    """

    def __init__(self, data, nr_samples):
        self.data = data
        self.nr_samples = nr_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
