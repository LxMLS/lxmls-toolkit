import torch
import random
import pickle
from torch.utils.data import Dataset
import numpy as np


class WeatherDataset(Dataset):
    """Dataset for training an auto regressive transformer on a sequence of weather/actions
    Input (observations): ['clean', 'clean', 'shop', 'walk', 'shop', 'read']
    Input (IDs): [0, 0, 2, 4, 2, 1]
    Output (states): ['sunny', 'rainy', 'rainy', 'sunny', 'snowy', 'sunny']
    Output (IDs): [7, 5, 5, 7, 6, 7]]
    Which we will feed into the transformer concatenated as:
    Input: [0, 0, 2, 4, 2, 1, 7, 5, 5, 7, 6]
    Output: [-1, -1, -1, -1, -1, 7, 5, 5, 7, 6, 7]
    where each observation and state are converted to an index ans -1 indicates "ignore",
    as the transformer is reading the input sequence but not predicting it.
    """

    def __init__(self, split, seq_len=6, num_instances=10000, proba=False):
        assert split in {'train', 'test'}
        self.split = split
        self.size = num_instances

        # Generate vocabulary
        self.obs, self.states = self.generate_voc()

        # Get HMM probabilities for dataset generation
        # We should work with a fixed proba, but there is a functoin for random generation
        if proba:
            self.proba = proba
        else:
            self.generate_random_proba()

        self.length = seq_len

    def __len__(self):
        return (self.size)

    def get_block_size(self):
        # the length of the sequence that will feed into transformer,
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def get_vocab_size(self):
        # Our vocabulary is the size of observation + states
        return (len(self.obs) + len(self.states))

    def generate_voc(self):
        """Generating vocabulary for the HMM model.
        Should not change that."""

        observations = ["walk", "shop", "clean", "tennis", "read"]
        states = ["sunny", "rainy", "snowy"]

        # Sort them alphabetically, just to be on the safe side
        observations.sort()
        states.sort()

        return (observations, states)

    # Dummy function for decoding
    def decode_obs(self, obs):
        return ([self.obs[i] for i in obs])

    # State IDs are offset by number of observations
    def decode_st(self, st):
        ofs = len(self.obs)
        return ([self.states[i - ofs] for i in st])

    def decode_seq(self, x, y):

        return (self.decode_obs(x), self.decode_st(y))

    # Dummy function for converting random logits to probabilities
    def logits_to_probs(self, logits):
        logits = np.array(
            logits
        )  # Convert the list to a numpy array for efficient calculations
        exp_logits = np.exp(
            logits)  # Apply the exponential function to each element
        probabilities = exp_logits / np.sum(
            exp_logits)  # Divide each element by the sum of all elements
        return probabilities.tolist(
        )  # Convert the numpy array back to a Python list

    # We should NOT use that.
    # Mostly for debugging purposes
    # The resulting dataset is almost unlearnable as it's randomly generated
    def generate_random_proba(self):

        # Generating a probability distribution for HMM
        self.proba = {}

        # Initial probabilities
        self.proba["initial"] = []

        # Generate random initial probabilities for each state
        for state in self.states:
            self.proba["initial"].append(random.random())

        # Convert to probabilities
        self.proba["initial"] = self.logits_to_probs(self.proba["initial"])

        # Transition probabilities
        self.proba["transition"] = []

        # Generate transition from state x to any other state
        for state in self.states:
            c_t_pr = []

            # Generate random tr probabilities for all states
            for state in self.states:
                c_t_pr.append(random.random())

            # N.B. we do NOT generate "Final" probabilities
            # We will generate a fixed length sequence instead
            # Lazy solution, I know...

            # Convert to probabilities
            c_t_pr = self.logits_to_probs(c_t_pr)

            self.proba["transition"].append(c_t_pr)

        # Emission probabilities
        self.proba["emission"] = []

        # Generate emission from state x to any observation
        for state in self.states:
            c_e_pr = []

            # Generate random em probabilities for all observations
            for obs in self.obs:
                c_e_pr.append(random.random())

            c_e_pr = self.logits_to_probs(c_e_pr)

            self.proba["emission"].append(c_e_pr)

    # Dummy function for sampling w.r.t probability
    def sample_p(self, p_l):
        items = np.arange(len(p_l))
        sample = np.random.choice(items, p=p_l)
        return sample

    def generate_seq(self):
        """Generating a random sequence given probas"""

        # Variable initialization
        eos = False
        c_s = 99
        x = []
        y = []

        while not eos:

            # Start of sequence
            if c_s == 99:
                # Sample from initial
                c_s = self.sample_p(self.proba["initial"])

            # Consecutive iterations

            # We generate until we get length of self length
            elif len(x) < self.length:
                # Sample from transition of last state
                c_s = self.sample_p(self.proba["transition"][c_s])

                # Generate emission

                # Note that we append the states as labels and observations as input
                y.append(c_s)
                x.append(self.sample_p(self.proba["emission"][c_s]))

            else:
                eos = True

        # We get the state ID by offseting their idx by the length of observations
        ofs = len(self.obs)
        y = [i + ofs for i in y]
        return (x, y)

    def __getitem__(self, idx):

        # use rejection sampling to generate an input example from the desired split
        while True:

            # Generate observation and its states
            obs, st = self.generate_seq()

            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(obs))
            inp_split = 'test' if h % 4 == 0 else 'train'  # designate 25% of examples as test
            if inp_split == self.split:
                break  # ok

        # concatenate the observation and labels
        cat = torch.cat((torch.LongTensor(obs), torch.LongTensor(st)), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length - 1] = -1
        return x, y