from os import path


class DependencyWriter():
    '''
    Dependency writer class
    '''
    def __init__(self):
        pass

    def save(self, language, heads_pred):
        '''Saves predicted dependency trees.''' 
        base_deppars_dir = path.join(path.dirname(__file__),"..","..","data","deppars")
        languages = ["danish","dutch","portuguese","english"]
        i = 0
        word_dict = {}
        pos_dict = {}
        feat_counts = {}
        if(language not in languages):
            print "Language does not exist: \"%s\": Available are: %s"%(language,languages)
            return

        ### Load test data
        n_toks = 0
        n_sents = 0
        conll_file = open(path.join(base_deppars_dir, language+"_test.conll"))
        conll_file_out = open(path.join(base_deppars_dir, language+"_test.conll.pred"), 'w')
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_toks = 0
                n_sents+=1
                conll_file_out.write("\n")
                continue
            fields = line.split("\t")

            fields[6] = "{0}".format(heads_pred[n_sents][n_toks+1])
            line_out = "\t".join(fields)
            n_toks+=1

            conll_file_out.write(line_out)
            conll_file_out.write("\n")
            
        conll_file_out.close()
        conll_file.close()



