# Ignore this file until it is mentioned in the guide
# This code should be used inside your mapper() method:

num_states = self.hmm.get_num_states() # Number of states.

yield 'log-likelihood', log_likelihood
for y in xrange(num_states):
    name_y = self.hmm.state_labels.get_label_name(y)
    for y in xrange(num_states):
        name_x = self.hmm.state_labels.get_label_name(y)
        yield 'transition %s %s' % (name_y, name_x), transition_counts[y,x]
    yield 'final ' + name_y, final_counts[y]
    yield 'initial ' + name_y, initial_counts[y]

for w in xrange(emission_counts):
    name_w = self.hmm.state_labels.get_label_name(w)
    if emission_counts[w].any():
        for s in xrange(num_states):
            name_s = self.hmm.state_labels.get_label_name(s)
            if emission_counts[w,s]:
                yield 'emission %s %s' % (name_w, name_s), emission_counts[w,s]

