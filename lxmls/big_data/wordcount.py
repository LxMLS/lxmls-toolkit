# Import the necessary libraries:
from mrjob.job import MRJob
from collections import Counter

class WordCount(MRJob):

    def mapper(self, _, doc):
        # Process the document
        c = dict(Counter(doc.split()))
        # Now, output the results
        for w, c in c.items():
            yield w, c

    def reducer(self, key, cs):
        yield key, sum(cs)


wc = WordCount()
wc.run()
