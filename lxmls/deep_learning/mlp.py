"""
Basic MLP class methods for parameters initialization, saving, loading
plotting
"""
import os
from six.moves import cPickle
import yaml
import numpy as np
from copy import deepcopy
from lxmls.deep_learning.utils import Model, glorot_weight_init


def load_parameters(parameter_file):
    """
    Load model
    """
    with open(parameter_file, 'rb') as fid:
        parameters = cPickle.load(fid)
    return parameters


def load_config(config_path):
    with open(config_path, 'r') as fid:
        config = yaml.load(fid)
    return config


def save_config(config_path, config):
    with open(config_path, 'w') as fid:
        yaml.dump(config, fid, default_flow_style=False)


def initialize_mlp_parameters(geometry, loaded_parameters=None,
                              random_seed=None):
    """
    Initialize parameters from geometry or existing weights
    """

    num_layers = len(geometry) - 1
    num_hidden_layers = num_layers - 1
    activation_functions = ['sigmoid']*num_hidden_layers + ['softmax']

    # Initialize random seed if not given
    if random_seed is None:
        random_seed = np.random.RandomState(1234)

    if loaded_parameters is not None:
        assert len(loaded_parameters) == num_layers, \
            "New geometry not matching model saved"

    parameters = []
    for n in range(num_layers):

        # Weights
        if loaded_parameters is not None:
            weight, bias = loaded_parameters[n]
            assert weight.shape == (geometry[n+1], geometry[n]), \
                "New geometry does not match for weigths in layer %d" % n
            assert bias.shape == (1, geometry[n+1]), \
                "New geometry does not match for bias in layer %d" % n

        else:
            weight = glorot_weight_init(
                (geometry[n], geometry[n+1]),
                activation_functions[n],
                random_seed
            )

            # Bias
            bias = np.zeros((1, geometry[n+1]))

        # Append parameters
        parameters.append([weight, bias])

    return parameters


def get_mlp_parameter_handlers(layer_index=None, is_bias=None, row=None,
                               column=None):
    """Returns the parameters of a multi-layer perceptron"""

    # Cast to integer
    is_bias = int(is_bias)

    def get_parameter(parameters):
        if is_bias:
            # bias
            return parameters[layer_index][is_bias][0][row]
        else:
            # weight
            return parameters[layer_index][is_bias][row, column]

    def set_parameter(parameters, parameter_value):
        if is_bias:
            # bias
            parameters[layer_index][is_bias][0][row] = parameter_value
        else:
            # weight
            parameters[layer_index][is_bias][row, column] = parameter_value
        return parameters

    return get_parameter, set_parameter


def get_mlp_loss_range(model, get_parameter, set_parameter, batch, span=10):

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


class MLP(Model):
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

        # MEMBER VARIABLES
        self.num_layers = len(config['geometry']) - 1
        self.config = config
        self.parameters = initialize_mlp_parameters(
            config['geometry'],
            loaded_parameters
        )

    def sanity_checks(self, config):

        model_folder = config.get('model_folder', None)

        assert bool(config is None) or bool(model_folder is None), \
            "Need to specify config, model_folder or both"

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
            cPickle.dump(self.parameters, fid, cPickle.HIGHEST_PROTOCOL)

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
