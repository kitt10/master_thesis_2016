# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_tool.nn_function
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Not a script, nor a lib. Just 'static' common functions used in other scripts and libs.
"""

import numpy as np
from termcolor import colored


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def output_layer(position, n_neurons):
    layer = np.zeros((n_neurons, 1))
    layer[position] = 1.0
    return layer


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, normed=False):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(str(x)) for x in labels]+[7]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            if normed:
                cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            else:
                cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print colored(str(cell), 'magenta'),
        print
