import codecs
from os import path


class Instance:
    """
    Instance class
    """

    def __init__(self):
        self.words = []
        self.pos = []
        self.heads = []


class DependencyReader:
    """
    Dependency reader class
    """

    def __init__(self):
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []

    def load(self, language):
        """Loads training and test data for dependency parsing."""
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []
        base_deppars_dir = path.join(path.dirname(__file__), "..", "..", "data", "deppars")
        languages = ["danish", "dutch", "portuguese", "english"]
        if language not in languages:
            print("Language does not exist: \"%s\": Available are: %s" % (language, languages))
            return

        # Create alphabet from training data
        n_sents = 0
        n_toks = 0
        word_id = 0
        pos_id = 0
        conll_file = codecs.open(path.join(base_deppars_dir, language + "_train.conll"), encoding='utf8')

        self.word_dict["__START__"] = word_id  # Start symbol
        word_id += 1
        self.word_dict["__STOP__"] = word_id  # Stop symbol
        word_id += 1
        self.pos_dict["__START__"] = pos_id  # Start symbol
        pos_id += 1
        self.pos_dict["__STOP__"] = pos_id  # Stop symbol
        pos_id += 1

        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents += 1
                continue
            fields = line.split("\t")
            n_toks += 1
            word = fields[1]
            pos = fields[3]
            if word not in self.word_dict:
                self.word_dict[word] = word_id
                word_id += 1
            if pos not in self.pos_dict:
                self.pos_dict[pos] = pos_id
                pos_id += 1
        conll_file.close()

        print("Number of sentences: {0}".format(n_sents))
        print("Number of tokens: {0}".format(n_toks))
        print("Number of words: {0}".format(word_id))
        print("Number of pos: {0}".format(pos_id))

        # Load training data
        self.train_instances = []
        inst = Instance()
        inst.words.append(self.word_dict["__START__"])
        inst.pos.append(self.pos_dict["__START__"])
        inst.heads.append(-1)
        conll_file = codecs.open(path.join(base_deppars_dir, language + "_train.conll"), encoding='utf8')
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents += 1
                self.train_instances.append(inst)
                inst = Instance()
                inst.words.append(self.word_dict["__START__"])
                inst.pos.append(self.pos_dict["__START__"])
                inst.heads.append(-1)
                continue
            fields = line.split("\t")

            word = fields[1]
            pos = fields[3]
            head = int(fields[6])

            if word not in self.word_dict:
                word_id = -1
            else:
                word_id = self.word_dict[word]
            if pos not in self.pos_dict:
                pos_id = -1
            else:
                pos_id = self.pos_dict[pos]

            inst.words.append(word_id)
            inst.pos.append(pos_id)
            inst.heads.append(head)

        conll_file.close()

        # Load test data
        self.test_instances = []
        inst = Instance()
        inst.words.append(self.word_dict["__START__"])
        inst.pos.append(self.pos_dict["__START__"])
        inst.heads.append(-1)
        conll_file = codecs.open(path.join(base_deppars_dir, language + "_test.conll"), encoding='utf8')
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents += 1
                self.test_instances.append(inst)
                inst = Instance()
                inst.words.append(self.word_dict["__START__"])
                inst.pos.append(self.pos_dict["__START__"])
                inst.heads.append(-1)
                continue
            fields = line.split("\t")

            word = fields[1]
            pos = fields[3]
            head = int(fields[6])

            if word not in self.word_dict:
                word_id = -1
            else:
                word_id = self.word_dict[word]
            if pos not in self.pos_dict:
                pos_id = -1
            else:
                pos_id = self.pos_dict[pos]

            inst.words.append(word_id)
            inst.pos.append(pos_id)
            inst.heads.append(head)  # gold heads

        conll_file.close()
