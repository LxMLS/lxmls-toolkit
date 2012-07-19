
from id_feature import IDFeatures

#######################
#### Feature Class
### Extracts features from a labeled corpus (only supported features are extracted
#######################
class ExtendedFeatures(IDFeatures):

    ##Add word tag pair
    def add_node_features(self,seq,pos,y,idx):
        x = seq.x[pos]
        y_name = self.dataset.int_to_tag[y]
        word = self.dataset.int_to_word[x]
        word = unicode(word)
        feat = "id:%s::%s"%(word,y_name)
        nr_feat = self.add_feature(feat)
        if(nr_feat != -1):
            idx.append(nr_feat)
        if( unicode.istitle(word)):
            feat = "uppercased::%s"%y_name
            nr_feat = self.add_feature(feat)
            if(nr_feat != -1):
                idx.append(nr_feat)
        if unicode.isdigit(word):
            feat = "number::%s"%y_name
            nr_feat = self.add_feature(feat)
            if(nr_feat != -1):
                idx.append(nr_feat)
        if (unicode.find(word,"-") != -1):
            feat = "hyphen::%s"%y_name
            nr_feat = self.add_feature(feat)
            if(nr_feat != -1):
                idx.append(nr_feat)
        ##Suffixes
        max_suffix = 3
        for i in xrange(max_suffix):
            if(len(word) > i+1):
                suffix = word[-(i+1):]
                feat = "suffix:%s::%s"%(suffix,y_name)
                nr_feat = self.add_feature(feat)
                if(nr_feat != -1):
                    idx.append(nr_feat)
        ##Prefixes
        max_prefix =3
        for i in xrange(max_prefix):
            if(len(word) > i+1):
                prefix = word[:i+1]
                feat = "prefix:%s::%s"%(prefix,y_name)
                nr_feat = self.add_feature(feat)
                if(nr_feat != -1):                        
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
        if(x not in self.dataset.word_counts or self.dataset.word_counts[x] <= 5):
            feat = "rare::%s"%(y_name)
            nr_feat = self.add_feature(feat)
            if(nr_feat != -1):                        
                idx.append(nr_feat)                        
        return idx

    
    
