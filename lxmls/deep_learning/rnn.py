"""
Basic MLP class methods for parameters initialization, saving, loading
plotting
"""
import os
from six.moves import cPickle as pickle
import yaml
import numpy as np
from copy import deepcopy
from lxmls.deep_learning.utils import Model


def load_parameters(parameter_file):
    """
    Load model
    """
    with open(parameter_file, 'rb') as fid:
        parameters = pickle.load(fid)
    return parameters


def load_config(config_path):
    with open(config_path, 'r') as fid:
        config = yaml.load(fid)
    return config


def save_config(config_path, config):
    with open(config_path, 'w') as fid:
        yaml.dump(config, fid, default_flow_style=False)


def initialize_rnn_parameters(input_size, embedding_size, hidden_size,
                              output_size, random_seed=None,
                              loaded_parameters=None):
    """
    Initialize parameters from geometry or existing weights
    """

    # Initialize random seed if not given
    if random_seed is None:
        random_seed = np.random.RandomState(1234)

    if loaded_parameters is not None:

        # LOAD MODELS

        assert len(loaded_parameters) == 4, \
            "New geometry not matching model saved"

        W_e, W_x, W_h, W_y = loaded_parameters

        # Note: Pytorch requires this shape order fro nn.Embedding()
        assert W_e.shape == (input_size, embedding_size), \
            "Embedding layer ze not matching saved model"
        assert W_x.shape == (hidden_size, embedding_size), \
            "Input layer ze not matching saved model"
        assert W_h.shape == (hidden_size, hidden_size), \
            "Hidden layer not matching saved model"
        assert W_y.shape == (output_size, hidden_size), \
            "Output layer size not matching saved model"

    else:

        # INITIALIZE

        # Input layer
        W_e = 0.01*random_seed.uniform(size=(input_size, embedding_size))
        # Input layer
        W_x = random_seed.uniform(size=(hidden_size, embedding_size))
        # Recurrent layer
        W_h = random_seed.uniform(size=(hidden_size, hidden_size))
        # Output layer
        W_y = random_seed.uniform(size=(output_size, hidden_size))

    return [W_e, W_x, W_h, W_y]


def get_rnn_parameter_handlers(layer_index=None, row=None, column=None):

    def get_parameter(parameters):
        # weight
        return parameters[layer_index][row, column]

    def set_parameter(parameters, parameter_value):
        # weight
        parameters[layer_index][row, column] = parameter_value
        return parameters

    return get_parameter, set_parameter


def get_rnn_loss_range(model, get_parameter, set_parameter, batch, span=10):

    # perturbation of  weight values
    perturbations = np.linspace(-span, span, 200)

    # Compute the loss when varying the study weight
    parameters = deepcopy(model.parameters)
    current_weight = float(get_parameter(parameters))
    loss_range = []
    old_parameters = list(model.parameters)
    for perturbation in perturbations:

        # Chage parameters
        model.parameters = set_parameter(
            parameters,
            current_weight + perturbation
        )

        # Compute loss
        perturbated_loss = model.cross_entropy_loss(
            batch['input'],
            batch['output']
        )
        loss_range.append(perturbated_loss)

    # Return to old parameters
    model.parameters = old_parameters

    weight_range = current_weight + perturbations
    return weight_range, loss_range


class RNN(Model):
    def __init__(self, **config):

        # CHECK THE PARAMETERS ARE VALID
        self.sanity_checks(config)

        # OPTIONAL MODEL LOADING
        model_folder = config.get('model_folder', None)
        if model_folder is not None:
            saved_config, loaded_parameters = self.load(model_folder)
            # Note that if a config is given this is used instead of the saved
            # one (must be consistent)
            if config is None:
                config = saved_config
        else:
            loaded_parameters = None

        # Class variables
        self.config = config
        self.parameters = initialize_rnn_parameters(
            config['input_size'],
            config['embedding_size'],
            config['hidden_size'],
            config['output_size'],
            loaded_parameters=loaded_parameters
        )

    def sanity_checks(self, config):

        model_folder = config.get('model_folder', None)

        assert bool(config is None) or bool(model_folder is None), \
            "Need to specify config, model_folder or both"

        if config is not None:
            pass

        if model_folder is not None:
            model_file = "%s/config.yml" % model_folder
            assert os.path.isfile(model_file), "Need to provide %s" % model_file

    def load(self, model_folder):
        """
        Load model
        """

        # Configuration un yaml format
        config_file = "%s/config.yml" % model_folder
        config = load_config(config_file)

        # Computation graph parameters as pickle file
        parameter_file = "%s/parameters.pkl" % model_folder
        loaded_parameters = load_parameters(parameter_file)

        return config, loaded_parameters

    def save(self, model_folder):
        """
        Save model
        """

        # Create folder if it does not exist
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        # Configuration un yaml format
        config_file = "%s/config.yml" % model_folder
        save_config(config_file, self.config)

        # Computation graph parameters as pickle file
        parameter_file = "%s/parameters.pkl" % model_folder
        with open(parameter_file, 'wb') as fid:
            pickle.dump(self.parameters, fid, pickle.HIGHEST_PROTOCOL)

    def plot_weights(self, show=True, aspect='auto'):
        """
        Plots the weights of the newtwork

        Use show = False to plot various models one after the other
        """
        import matplotlib.pyplot as plt
        plt.figure()
        for n in range(self.n_layers):

            # Get weights and bias
            weight, bias = self.parameters[n]

            # Plot them
            plt.subplot(2, self.n_layers, n+1)
            plt.imshow(weight, aspect=aspect, interpolation='nearest')
            plt.title('Layer %d Weight' % n)
            plt.colorbar()
            plt.subplot(2, self.n_layers, self.n_layers+(n+1))
            plt.plot(bias)
            plt.title('Layer %d Bias' % n)
            plt.colorbar()

        if show:
            plt.show()
