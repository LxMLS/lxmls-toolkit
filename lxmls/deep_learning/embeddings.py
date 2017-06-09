'''
Utilities to handle embeddings
'''
import os
import numpy as np
from six.moves import urllib
from lxmls import data

def download_embeddings(embbeding_name, target_file):
    '''
    Downloads file through http with progress report

    Obtained in stack overflow:
    http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http
    -using-python
    '''

    # Embedding download URLs
    if embbeding_name == 'senna_50':
        # senna_50 embeddings
        source_url = 'http://lxmls.it.pt/2015/wp-content/uploads/2015/senna_50'
    else:
        raise ValueError("I do not have embeddings %s for download"
                           % embbeding_name)

    target_file_name = os.path.basename(data.find('senna_50'))
    u = urllib.request.urlopen(source_url)
    with open(target_file, 'wb') as f:
        meta         = u.info()
        file_size    = int(meta.getheaders("Content-Length")[0])
        file_size_dl = 0
        block_sz     = 8192
        print("Downloading: %s Bytes: %s" % (target_file_name, file_size))
        while True:
            text_buffer = u.read(block_sz)
            if not text_buffer:
                break
            file_size_dl += len(text_buffer)
            f.write(text_buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl,
                                           file_size_dl*100./file_size)
            status = status + chr(8)*(len(status)+1)
            print(status)
    print("")

def extract_embeddings(embedding_path, word_dict):
    '''
    Given embeddings in text form and a word dictionary construct embedding
    matrix. Words with no embedding get initialized to random.
    '''
    with open(embedding_path) as fid:
        for i, line in enumerate(fid.readlines()):
            # Initialize
            if i == 0:
                 N    = len(line.split()[1:])
                 E    = np.random.uniform(size=(N, len(word_dict)))
                 n    = 0
            word = line.split()[0].lower()
            if word[0].upper() + word[1:] in word_dict:
                idx        = word_dict[word[0].upper() + word[1:]]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            elif word in word_dict:
                idx        = word_dict[word]
                E[:, idx]  = np.array(line.strip().split()[1:]).astype(float)
                n         += 1
            print("\rGetting embeddings for the vocabulary %d/%d" % \
                (n, len(word_dict)))
    OOV_perc =  (1-n*1./len(word_dict))*100
    print("\n%2.1f%% missing embeddings, set to random" % OOV_perc)
    return E
