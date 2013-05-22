#######
This directory contains code for POS-Tagging

Steps
1 - Get the tagged corpus for decoding. For now we are going to use the brown corpus that ships with NLTK. In the future we can increase the corpus
2 - Map the tagset, I am moving from browns 90 categories to the universal 12 tags tagset of Slav Petrov. This will make the code much faster, it will allow to use different corpus in the future. If we realize we need more tags we can then use a different method to distinguish categories.


3 - Feature extraction from the corpus
4 - Train a model this will be a sequence model from ou Machine Learning Toolkit ( I will start to use the one I developed for the summer school and this will be improved with time).
5 - Save the model.

6 - Tag new sentences (this should be fast

In the future this module might be extended to deal with paralization and so on. 


Todo:
 
- Remove build edge features, pass transition matrix instead even if
  we have to account for special transition features for final state

- Test performance, accuracy for increasing dataset sizes on wsj

- Why is viterbi returning list of lists

- Look at precomputed features for gold in feture class as a speed up

- Test on brown, lookout for different domains and their impact on
  perfomance

- Test on test set that does not contain ids, document

- Incorporate into the pipeline
