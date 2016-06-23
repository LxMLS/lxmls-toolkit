import id_feature as idf


class BasicFeatures(idf.IDFeatures):

    def __init__(self, dataset):
        idf.IDFeatures.__init__(self, dataset)

    # def add_next_word_context_feature(self,next_word,tag,idx):
    #     feat = "next_word:%s::%s"%(next_word,tag)
    #     nr_feat = self.add_feature(feat)
    #     idx.append(nr_feat)
    #     return idx

    # def add_prev_word_context_feature(self,prev_word,tag,idx):
    #     feat = "prev_word:%s::%s"%(prev_word,tag)
    #     nr_feat = self.add_feature(feat)
    #     idx.append(nr_feat)
    #     return idx

    def add_node_feature(self, seq, pos, y, idx):
        x = seq.x[pos]
        word = self.dataset.int_to_word[x]
        if self.dataset.word_counts[x] > 5:
            y_name = self.dataset.int_to_pos[y]
            word = self.dataset.int_to_word[x]
            feat = "id:%s::%s" % (word, y_name)
            nr_feat = self.add_feature(feat)
            idx.append(nr_feat)
        else:
            # Check for upercase
            if not unicode.islower(word):
                feat = "upercased::%s" % y
                nr_feat = self.add_feature(feat)
                idx.append(nr_feat)
            # Check for number
            if not unicode.isalpha(word):
                feat = "number::%s" % y
                nr_feat = self.add_feature(feat)
                idx.append(nr_feat)

            # Check for number
            if unicode.find(word, "-") != -1:
                feat = "hyphen::%s" % y
                nr_feat = self.add_feature(feat)
                idx.append(nr_feat)

            # Suffixes
            max_suffix = 4
            for i in xrange(max_suffix):
                if len(word) > i+1:
                    suffix = word[-(i+1):]
                    feat = "suffix:%s::%s" % (suffix, y)
                    nr_feat = self.add_feature(feat)
                    idx.append(nr_feat)
            # Prefixes
            max_prefix = 4
            for i in xrange(max_prefix):
                if len(word) > i+1:
                    prefix = word[:i+1]
                    feat = "prefix:%s::%s" % (prefix, y)
                    nr_feat = self.add_feature(feat)
                    idx.append(nr_feat)
        return idx
