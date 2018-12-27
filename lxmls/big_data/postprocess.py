import math
import pdb


def load_counts(ifile):
    counts = {}
    total_kmers = 0.0
    with open(ifile) as iinput:
        for line in iinput:
            word, count = line.strip().split('\t')
            word = word[1:-1]
            count = int(count)
            counts[word] = float(count + 1)  # the +1 implements add-one smoothing
            total_kmers += float(count + 1)
    return counts, total_kmers


def score(counts_pt, total_trimers_pt, counts_en, total_trimers_en, test_sentence):
    val = 0.
    for i in range(len(test_sentence)-3):
        tri = test_sentence[i:i+3]
        tri_pt = counts_pt.get(tri, 1.0)  # this will attempt to get counts from the dictionary; if it fails, it will return 1.0
        log_prob_tri_pt = math.log10(tri_pt / total_trimers_pt)
        tri_en = counts_en.get(tri, 1.0)
        log_prob_tri_en = math.log10(tri_en / total_trimers_en)

        val += log_prob_tri_pt - log_prob_tri_en

    if val >= 0:
        language = "PT"
    else:
        language = "EN"
    if abs(val) >= 5:
        print("This is a", language, "sentence.")
    else:
        print("This seems to be a", language, "sentence, but I'm not sure.")
    print("Log-ratio:", abs(val))


counts_pt, total_trimers_pt = load_counts('pt.counts.txt')
counts_en, total_trimers_en = load_counts('en.counts.txt')

while True:
    test_sentence = input("Type a test sentence and press ENTER:\n")
    if not test_sentence:
        break
    score(counts_pt, total_trimers_pt, counts_en, total_trimers_en, test_sentence)
