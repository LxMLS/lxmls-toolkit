# Import the necessary libraries:
from mrjob.job import MRJob
import re
import pdb


class TrimerCount(MRJob):

    def mapper(self, _, doc):
        c = {}
        # Process the document
        for i in range(len(doc)-3):
            w = doc[i:i+3]
            if w in c:
                c[w] += 1
            else:
                c[w] = 1

        # Now, output the results
        for w, c in c.items():
            yield w, c

    def reducer(self, key, cs):
        yield key, sum(cs)


TrimerCount.run()
