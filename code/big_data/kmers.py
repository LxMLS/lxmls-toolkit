# Import the necessary libraries:
from mrjob.job import MRJob
import re

# Check whether the word is a real word
def _is_valid_word(w):
    return re.search(r'\W', w) is None

class TrimerCount(MRJob):
    def mapper(self, _, doc):
        c = {}
        # Process the document
        for i in xrange(len(doc)-3):
            w = doc[i:i+3]
            if _is_valid_word(w):
                if w in c:
                    c[w] += 1
                else:
                    c[w] = 1

        # Now, output the results
        for w,c in c.items():
            yield w,c

    def reducer(self, key, cs):
        yield key, sum(cs)

TrimerCount.run()
