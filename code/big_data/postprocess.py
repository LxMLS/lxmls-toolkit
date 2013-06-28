def load_counts(ifile):
    counts = {}
    with open(ifile) as input:
        for line in input:
            word, count = line.strip().split()
            word = word[1:-1]
            counts[word] = float(count)
    return counts

def score(counts_pt, counts_en, test):
    val = 1.
    for i in xrange(len(test)-3):
        tri = test[i:i+3]
        tri_pt = counts_pt.get(tri, 1.)
        tri_en = counts_en.get(tri, 1.)
        val *= tri_pt/tr_en
    return val

counts_pt = load_counts('output.pt.txt')
counts_en = load_counts('output.en.txt')

while True:
    test = raw_input("Type a test sentence? ")
    if not test: break
    print score(counts_pt, counts_en, test)

