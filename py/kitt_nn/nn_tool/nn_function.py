# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_tool.nn_function
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Not a script, nor a lib. Just 'static' common functions used in other scripts and libs.
"""

import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def output_layer(position, n_neurons):
    layer = np.zeros((n_neurons, 1))
    layer[position] = 1.0
    return layer