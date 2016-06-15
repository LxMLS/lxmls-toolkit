from sequences.label_dictionary import *


class AbstractFeatureClass(object):
    """ Defines an abstract feature class used to
    build the node and edge potentials.

       All feature classes should implement this class
    """

    def __init__(self, dataset):
        """dataset is a sequence list."""
        self.feature_dict = LabelDictionary()
        #        self.feature_names = []
        #        self.nr_feats = 0
        self.feature_list = []

        self.add_features = False
        self.dataset = dataset

        # Speed up
        self.node_feature_cache = {}
        self.initial_state_feature_cache = {}
        self.final_state_feature_cache = {}
        self.edge_feature_cache = {}

    #    def build_features(self):
    #        '''
    #        Generic function to build features for a given dataset.
    #        Iterates through all sentences in the dataset and extracts its features,
    #        saving the node/edge features in feature list.
    #        '''
    #        self.add_features = True
    #        for seq in self.dataset.sequence_list.seq_list:
    #           seq_node_features,seq_edge_features = self.get_seq_features(seq)
    #           self.feature_list.append([seq_node_features,seq_edge_features])
    #        self.nr_feats = len(self.feature_names)
    #        self.add_features = False

    def get_num_features(self):
        return len(self.feature_dict)

    def build_features(self):
        """
        Generic function to build features for a given dataset.
        Iterates through all sentences in the dataset and extracts its features,
        saving the node/edge features in feature list.
        """
        self.add_features = True
        for sequence in self.dataset.seq_list:
            initial_features, transition_features, final_features, emission_features = \
                self.get_sequence_features(sequence)
            self.feature_list.append([initial_features, transition_features, final_features, emission_features])
        # self.nr_feats = len(self.feature_names)
        self.add_features = False

    def get_transition_features(self, sequence, position, tag_id, prev_tag_id):
        """
        Returns the edge features for position, for
        previous tag_id and tag_id.

        Note for a sentence of lenght N pos can go from 0 to N.
        Where:
        0 - Takes the initial transition features
        N - Takes the final transition features
        """
        raise NotImplementedError

    def get_initial_features(self, sequence, tag_id):
        raise NotImplementedError

    def get_final_features(self, sequence, prev_tag_id):
        raise NotImplementedError

    def get_emission_features(self, sequence, position, tag_id):
        """
        Returns all features for a node at position with tag_id
        """
        raise NotImplementedError

# def save_features(self,file):
#        fn = open(file,"w")
#        for feat_nr,feat in enumerate(self.feature_names):
#            fn.write("%i\t%s\n"%(feat_nr,feat))
#        fn.close()
#
#
#    ###########
#    # Loads all features form a file
#    ###########
#    def load_features(self,file,dataset):
#        fn = open(file)
#        self.feature_names = []
#        self.feature_dic = {}
#        for line in fn:
#            feat_nr,feat = line.strip().split("\t")
#            self.feature_names.append(feat)
#            self.feature_dic[feat] = int(feat_nr)
#        self.feature_list = []
#        self.nr_feats = len(self.feature_names)
#        self.add_features = False
#        self.dataset = dataset
#        fn.close()
#        #Speed up
#        self.node_feature_cache = {}
#        self.initial_state_feature_cache = {}
#        self.edge_feature_cache = {}
#        self.final_edge_feature_cache = {}
