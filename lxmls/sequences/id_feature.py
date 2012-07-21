
from .abstract_feature_class import AbstractFeatureClass


#################
### Replicates teh same features as the HMM
### One for word/tag and tag/tag pair
#################
class IDFeatures(AbstractFeatureClass):
    '''
        Base class to extract features from a particular dataset.

        feature_dic --> Dictionary of all existing features maps feature_name (string) --> feature_id (int) 
        feture_names --> List of feature names. Each position is the feature_id and contains the feature name
        nr_feats --> Total number of features
        feature_list --> For each sentence in the corpus contains a pair of node feature and edge features
        dataset --> The original dataset for which the features were extracted

        Caches (for speedup):
        initial_state_feature_cache -->
        node_feature_cache -->
        edge_feature_cache -->
        final_state_feature_cache -->
        
    '''

        

    def get_seq_features(self,seq):
        '''
        Returns the features for a given sequence.
        For a sequence of size N returns:
        Node_feature a list of size N. Each entry contains the node potentials for that position.
        Edge_features a list of size N+1.
        - Entry 0 contains the initial features
        - Entry N contains the final features
        - Entry i contains entries mapping the transition from i-1 to i.
        '''
        seq_node_features = []
        seq_edge_features = []
        ## Take care of first position
        init_idx = []
        node_idx = []
        node_idx = self.add_node_features(seq,0,seq.y[0],node_idx)
        init_idx = self.add_init_state_features(seq,seq.y[0],init_idx)
        seq_node_features.append(node_idx)
        seq_edge_features.append(init_idx)
        ## Take care of middle positions
        for i,tag in enumerate(seq.y[1:]):
            idx = []
            edge_idx = []
            j = i+1
            #print i,j
            prev_tag = seq.y[j-1]
            edge_idx = self.add_edge_features(seq,j,tag,prev_tag,edge_idx)            
            idx = self.add_node_features(seq,j,tag,idx)
            seq_node_features.append(idx)
            seq_edge_features.append(edge_idx)
        ## Take care of final position
        final_idx = []
        tag = seq.y[-1]
        final_idx = self.add_final_state_features(seq,tag,final_idx)
        seq_edge_features.append(final_idx)
        return seq_node_features,seq_edge_features


    #f(t,y_t,X)
    # Add the word identity and if position is
    # the first also adds the tag position
    def get_node_features(self,seq,pos,y):
        all_feat = []
        x = seq.x[pos]
        if(x not in self.node_feature_cache):
            self.node_feature_cache[x] = {}
        if(y not in self.node_feature_cache[x]):
            node_idx = []
            node_idx = self.add_node_features(seq,pos,y,node_idx)
            self.node_feature_cache[x][y] = node_idx
        idx = self.node_feature_cache[x][y]
        all_feat = idx[:]
        return all_feat



    #f(t,y_t,y_(t-1),X)
    ##Speed up of code
    def get_edge_features(self,seq,pos,y,y_prev):
        if(pos == 0):
           if(y not in self.initial_state_feature_cache):
               edge_idx = []
               edge =  self.add_init_state_features(seq,y,edge_idx)
               self.initial_state_feature_cache[y] = edge_idx
           return self.initial_state_feature_cache[y]
        elif(pos == len(seq.x)):
            if(y_prev not in self.final_state_feature_cache):
                edge_idx = []
                edge = self.add_final_state_features(seq,y_prev,edge_idx)            
                self.final_state_feature_cache[y_prev] = edge_idx
            return self.final_state_feature_cache[y_prev]
        else:
            if(y not in self.edge_feature_cache):
                self.edge_feature_cache[y]={}
            if(y_prev not in self.edge_feature_cache[y]): 
                edge_idx = []
                edge = self.add_edge_features(seq,pos,y,y_prev,edge_idx)            
                self.edge_feature_cache[y][y_prev] = edge_idx
            return self.edge_feature_cache[y][y_prev]


    def add_init_state_features(self,seq,y,init_idx):
        y_name = self.dataset.int_to_tag[y]
        feat = "init_tag:%s"%(y_name)
        nr_feat = self.add_feature(feat)
        if(nr_feat != -1):
            init_idx.append(nr_feat)
        return init_idx


    def add_final_state_features(self,seq,y_prev,final_idx):
        y_name = self.dataset.int_to_tag[y_prev]
        feat = "final_prev_tag:%s"%(y_name)
        nr_feat = self.add_feature(feat)
        if(nr_feat != -1):
            final_idx.append(nr_feat)
        return final_idx


        ##Add word tag pair
    def add_node_features(self,seq,pos,y,idx):
        x = seq.x[pos]
        y_name = self.dataset.int_to_tag[y]
        word = self.dataset.int_to_word[x]
        feat = "id:%s::%s"%(word,y_name)
        nr_feat = self.add_feature(feat)
        if(nr_feat != -1):
            idx.append(nr_feat)
        return idx




    def add_edge_features(self,seq,pos,y,y_prev,edge_idx):
        """ Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        #print "Adding edge feature for pos:%i y:%i y_prev%i seq_len:%i"%(pos,y,y_prev,len(seq.x))
        if(pos == len(seq.x)):
            y_prev_name = self.dataset.int_to_tag[y_prev]
            feat = "final_tag:%s"%(y_prev_name)
        else:
            y_name = self.dataset.int_to_tag[y]
            y_prev_name = self.dataset.int_to_tag[y_prev]
            feat = "prev_tag:%s::%s"%(y_prev_name,y_name)
        if(pos == len(seq.x)):
            print "adding feature %s"%feat
        nr_feat = self.add_feature(feat)
        if(nr_feat != -1):
            edge_idx.append(nr_feat)
        if(pos == len(seq.x)):
            print "Returning edges"
            print edge_idx
        return edge_idx


    def add_feature(self,feat):
        """
        Builds a dictionary of feature name to feature id
        If we are at test time and we don't have the feature
        we return -1.
        """
        
        # if(self.add_features == False):
        #     print feat
        if(feat in self.feature_dic):
            return self.feature_dic[feat]
        if not self.add_features:
            return -1
        nr_feat = len(self.feature_dic.keys())
        #print "Adding feature %s %i"%(feat,nr_feat)
        self.feature_dic[feat] = nr_feat
        self.feature_names.append(feat)
        return nr_feat

    ###### Printing functions ##############


    def get_sequence_feat_str(self,seq):
        seq_nr = seq.nr
        node_f_list = self.feature_list[seq_nr][0]
        edge_f_list = self.feature_list[seq_nr][1]

        word = seq.x[0]
        word_n = self.dataset.int_to_word[word]
        tag = seq.y[0]
        tag_n = self.dataset.int_to_tag[tag]
        txt = ""
        for i,tag in enumerate(seq.y):
            word = seq.x[i]
            word_n = self.dataset.int_to_word[word]
            tag_n = self.dataset.int_to_tag[tag]
            txt += "%i %s/%s NF: "%(i,word_n,tag_n)
            for nf in node_f_list[i]:
                txt+="%s "%self.feature_names[nf]
            if(edge_f_list[i] != []):
                txt += "EF: "
                for nf in edge_f_list[i]:
                    txt+="%s "%self.feature_names[nf]
            txt +="\n"
        return txt

    def print_sequence_features(self,seq):
        txt = ""
        for i,tag in enumerate(seq.y):
            word = seq.x[i]
            word_n = self.dataset.int_to_word[word]
            tag_n = self.dataset.int_to_tag[tag]
            txt += "%i %s/%s NF: "%(i,word_n,tag_n)
            if(i > 0):
                prev_tag = seq.y[i-1]
                edge_f_list = self.get_edge_features(seq,i,tag,prev_tag)
            else:
                edge_f_list = self.get_edge_features(seq,i,tag,-1)
            node_f_list = self.get_node_features(seq,i,tag)
            for nf in node_f_list:
                txt+="%s "%self.feature_names[nf]
            if(edge_f_list != []):
                txt += "EF: "
                for nf in edge_f_list:
                    txt+="%s "%self.feature_names[nf]
            txt +="\n"
        ## Add last position
        pos = len(seq.x)
        prev_tag = seq.y[pos-1]
        txt += "%i %s/%s NF: EF: "%(pos,-1,-1)
        edge_f_list = self.get_edge_features(seq,pos,-1,prev_tag)
        for nf in edge_f_list:
            txt+="%s "%self.feature_names[nf]
        txt +="\n"
        return txt

