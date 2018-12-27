# ----------
# Feature extraction
# ----------


class ExtendedFeatures:

    def __init__(self, dataset):
        self.feature_dic = {}
        self.feature_names = []
        self.nr_feats = 0
        self.feature_list = []
        self.add_features = False
        self.dataset = dataset
        # Speed up
        self.node_feature_cache = {}
        self.initial_state_feature_cache = {}
        self.edge_feature_cache = {}
        self.final_edge_feature_cache = {}

    def build_features(self):
        self.add_features = True
        for seq in self.dataset.train.seq_list:
            seq_node_features, seq_edge_features = self.get_seq_features(seq)
            self.feature_list.append([seq_node_features, seq_edge_features])
        self.nr_feats = len(self.feature_names)
        self.add_features = False

    def get_seq_features(self, seq):
        seq_node_features = []
        seq_edge_features = []
        # Take care of first position
        idx = []
        idx = self.add_node_features(seq, 0, seq.y[0], idx)
        idx = self.add_init_state_features(seq, 0, seq.y[0], idx)
        seq_node_features.append(idx)
        seq_edge_features.append([])
        for i, tag in enumerate(seq.y[1:]):
            idx = []
            edge_idx = []
            j = i + 1
            # print i,j
            prev_tag = seq.y[j-1]
            edge_idx = self.add_edge_features(seq, j, tag, prev_tag, edge_idx)
            idx = self.add_node_features(seq, j, tag, idx)
            seq_node_features.append(idx)
            seq_edge_features.append(edge_idx)
        return seq_node_features, seq_edge_features

    # Add word tag pair
    def add_node_features(self, seq, pos, y, idx):
        x = seq.x[pos]
        y_name = self.dataset.int_to_pos[y]
        word = self.dataset.int_to_word[x]
        feat = "id:%s::%s" % (word, y_name)
        nr_feat = self.add_feature(feat)
        if nr_feat != -1:
            idx.append(nr_feat)
        if unicode.istitle(word):
            feat = "uppercased::%s" % y_name
            nr_feat = self.add_feature(feat)
            if nr_feat != -1:
                idx.append(nr_feat)
            if unicode.isdigit(word):
                feat = "number::%s" % y_name
                nr_feat = self.add_feature(feat)
                if nr_feat != -1:
                    idx.append(nr_feat)
            if unicode.find(word, "-") != -1:
                feat = "hyphen::%s" % y_name
                nr_feat = self.add_feature(feat)
                if nr_feat != -1:
                    idx.append(nr_feat)
            # Suffixes
            max_suffix = 3
            for i in xrange(max_suffix):
                if len(word) > i+1:
                    suffix = word[-(i+1):]
                    feat = "suffix:%s::%s" % (suffix, y_name)
                    nr_feat = self.add_feature(feat)
                    if nr_feat != -1:
                        idx.append(nr_feat)
            # Prefixes
            max_prefix = 3
            for i in xrange(max_prefix):
                if len(word) > i+1:
                    prefix = word[:i+1]
                    feat = "prefix:%s::%s" % (prefix, y_name)
                    nr_feat = self.add_feature(feat)
                    if nr_feat != -1:
                        idx.append(nr_feat)
            # if(pos > 0):
            #     prev_word = seq.x[pos-1]
            #     if(self.dataset.word_counts[prev_word] > 5):
            #         prev_word_name = self.dataset.int_to_word[prev_word]
            #         feat = "prev_word:%s:%s"%(prev_word_name,y_name)
            #         nr_feat = self.add_feature(feat)
            #         if(nr_feat != -1):
            #             idx.append(nr_feat)
            # if(pos < len(seq.x) -1):
            #     next_word = seq.x[pos+1]
            #     if(self.dataset.word_counts[next_word] > 5):
            #         next_word_name = self.dataset.int_to_word[next_word]
            #         feat = "next_word:%s:%s"%(next_word_name,y_name)
            #         nr_feat = self.add_feature(feat)
            #         if(nr_feat != -1):
            #             idx.append(nr_feat)
            if self.dataset.word_counts[x] <= 5:
                feat = "rare::%s" % y_name
                nr_feat = self.add_feature(feat)
                if nr_feat != -1:
                    idx.append(nr_feat)

        return idx

    # f(t,y_t,X)
    # Add the word identity and if position is
    # the first also adds the tag position
    def get_node_features(self, seq, pos, y):
        all_feat = []
        x = seq.x[pos]
        if x not in self.node_feature_cache:
            self.node_feature_cache[x] = {}
        if y not in self.node_feature_cache[x]:
            node_idx = []
            node_idx = self.add_node_features(seq, pos, y, node_idx)
            # node_idx = filter (lambda a: a != -1, node_idx)
            self.node_feature_cache[x][y] = node_idx
        idx = self.node_feature_cache[x][y]
        all_feat = idx[:]
        # print idx
        if pos == 0:
            if y not in self.initial_state_feature_cache:
                init_idx = []
                init_idx = self.add_init_state_features(seq, pos, y, init_idx)
                self.initial_state_feature_cache[y] = init_idx
            all_feat.extend(self.initial_state_feature_cache[y])
        # print "before init"
        # print idx
        # print "after init"
        return all_feat

    def add_init_state_features(self, seq, pos, y, init_idx):
        y_name = self.dataset.int_to_pos[y]
        feat = "init_tag:%s" % y_name
        nr_feat = self.add_feature(feat)
        if nr_feat != -1:
            init_idx.append(nr_feat)
        return init_idx

    def add_edge_features(self, seq, pos, y, y_prev, edge_idx):
        # print "Adding edge feature for pos:%i y:%i y_prev%i seq_len:%i"%(pos,y,y_prev,len(seq.x))
        y_name = self.dataset.int_to_pos[y]
        y_prev_name = self.dataset.int_to_pos[y_prev]
        if pos == len(seq.x)-1:
            feat = "last_prev_tag:%s::%s" % (y_prev_name, y_name)
        else:
            feat = "prev_tag:%s::%s" % (y_prev_name, y_name)
        nr_feat = self.add_feature(feat)
        if nr_feat != -1:
            edge_idx.append(nr_feat)
        return edge_idx

    # f(t,y_t,y_(t-1),X)
    # Speed up of code
    def get_edge_features(self, seq, pos, y, y_prev):
        # print "Getting edge feature for pos:%i y:%i y_prev%i seq_len:%i"%(pos,y,y_prev,len(seq.x))
        # print "edge cache"
        # print self.edge_feature_cache
        # print "Final edge cache"
        # print self.final_edge_feature_cache
        if pos == len(seq.x)-1:
            if y not in self.final_edge_feature_cache:
                self.final_edge_feature_cache[y] = {}
            if y_prev not in self.final_edge_feature_cache[y]:
                edge_idx = []
                edge = self.add_edge_features(seq, pos, y, y_prev, edge_idx)
                self.final_edge_feature_cache[y][y_prev] = edge_idx
            return self.final_edge_feature_cache[y][y_prev]
        else:
            if y not in self.edge_feature_cache:
                self.edge_feature_cache[y] = {}
            if y_prev not in self.edge_feature_cache[y]:
                edge_idx = []
                edge = self.add_edge_features(seq, pos, y, y_prev, edge_idx)
                self.edge_feature_cache[y][y_prev] = edge_idx
            return self.edge_feature_cache[y][y_prev]

    def add_feature(self, feat):
        # if(self.add_features == False):
        #     print feat
        if feat in self.feature_dic:
            return self.feature_dic[feat]
        if not self.add_features:
            return -1
        nr_feat = len(self.feature_dic.keys())
        # print "Adding feature %s %i"%(feat,nr_feat)
        self.feature_dic[feat] = nr_feat
        self.feature_names.append(feat)
        return nr_feat

    def get_sequence_feat_str(self, seq):
        seq_nr = seq.nr
        node_f_list = self.feature_list[seq_nr][0]
        edge_f_list = self.feature_list[seq_nr][1]

        word = seq.x[0]
        word_n = self.dataset.int_to_word[word]
        tag = seq.y[0]
        tag_n = self.dataset.int_to_pos[tag]
        txt = ""
        for i, tag in enumerate(seq.y):
            word = seq.x[i]
            word_n = self.dataset.int_to_word[word]
            tag_n = self.dataset.int_to_pos[tag]
            txt += "%i %s/%s NF: " % (i, word_n, tag_n)
            for nf in node_f_list[i]:
                txt += "%s " % self.feature_names[nf]
            if edge_f_list[i] != []:
                txt += "EF: "
                for nf in edge_f_list[i]:
                    txt += "%s " % self.feature_names[nf]
            txt += "\n"
        return txt

    def print_sequence_features(self, seq):
        txt = ""
        for i, tag in enumerate(seq.y):
            word = seq.x[i]
            word_n = self.dataset.int_to_word[word]
            tag_n = self.dataset.int_to_pos[tag]
            txt += "%i %s/%s NF: " % (i, word_n, tag_n)
            prev_tag = seq.y[i-1]
            if i > 0:
                edge_f_list = self.get_edge_features(seq, i, tag, prev_tag)
            else:
                edge_f_list = []
            node_f_list = self.get_node_features(seq, i, tag)
            for nf in node_f_list:
                txt += "%s " % self.feature_names[nf]
            if edge_f_list != []:
                txt += "EF: "
                for nf in edge_f_list:
                    txt += "%s " % self.feature_names[nf]
            txt += "\n"

        return txt
