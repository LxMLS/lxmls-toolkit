# Import the necessary libraries:
from mrjob.job import MRJob


class WordCount(MRJob):

    def mapper(self, _, doc):
        c = {}
        # Process the document
        for w in doc.split():
            if w in c:
                c[w] += 1
            else:
                c[w] = 1

        # Now, output the results
        for w, c in c.items():
            yield w, c

    def reducer(self, key, cs):
        yield key, sum(cs)


wc = WordCount()
wc.run()
