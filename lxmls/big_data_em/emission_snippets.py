# The following code should be part of your mapper_final function

# num_states = self.hmm.get_num_states() # Number of states.
# num_observations = self.hmm.get_num_observations() # Number of observation symbols.
#
# yield 'log-likelihood', self.log_likelihood
# for y in xrange(num_states):
#     name_y = self.hmm.state_labels.get_label_name(y)
#     for s in xrange(num_states):
#         name_s = self.hmm.state_labels.get_label_name(s)
#         yield 'transition %s %s' % (name_y, name_s), self.transition_counts[y,s]
#     yield 'final ' + name_y, self.final_counts[y]
#     yield 'initial ' + name_y, self.initial_counts[y]
#
# for w in xrange(num_observations):
#     name_w = self.hmm.observation_labels.get_label_name(w)
#     if self.emission_counts[w].any():
#         for s in xrange(num_states):
#             name_s = self.hmm.state_labels.get_label_name(s)
#             if self.emission_counts[w,s]:
#                 yield 'emission %s %s' % (name_w, name_s), self.emission_counts[w,s]
