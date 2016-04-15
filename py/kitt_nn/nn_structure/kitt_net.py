#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_structure.kitt_net
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Desc
"""


import numpy as np
from kitt_neuron import *
from kitt_synapse import *
from kitt_nn.nn_tool.nn_function import output_layer


class NeuralNet(object):

    def __init__(self, program, name, structure):
        self.program = program
        self.name = name
        self.structure = structure
        self.ol_index = len(self.structure) - 1         # output layer index
        self.l_indexes = range(self.ol_index + 1)       # layers indexes [0, 1, ..., ol_index]
        self.neuronsG = list()                          # [gind]
        self.neuronsLP = dict()                         # [layer_index][position]
        self.synapsesG = list()                         # [gind]
        self.synapsesNN = dict()                        # [neuron_from][neuron_to]

        # Init net parameters randomly as np.arrays
        self.weights = [np.random.randn(n, m) for m, n in zip(self.structure[:-1], self.structure[1:])]
        self.biases = [np.random.randn(n, 1) for n in self.structure[1:]]

        # Coefficients for synapses (useful when removing them)
        self.synapses_exist = [np.ones((n, m)) for m, n in zip(self.structure[:-1], self.structure[1:])]

        # Create units and connect net
        self.create_neurons()
        self.connect_net_fully_ff()

        # Learning algorithm
        self.learning = None
        self.labels = None

    def create_neurons(self):
        # Input neurons
        self.neuronsLP[0] = list()
        for i in range(self.structure[0]):
            InputNeuron(self, 0)

        # Hidden neurons
        for layer_ind in self.l_indexes[1:-1]:
            self.neuronsLP[layer_ind] = list()
            for i in range(self.structure[layer_ind]):
                HiddenNeuron(self, layer_ind)

        # Output neurons
        self.neuronsLP[self.ol_index] = list()
        for i in range(self.structure[self.ol_index]):
            OutputNeuron(self, self.ol_index)

    def connect_net_fully_ff(self):
        for layer_ind in self.l_indexes[:-1]:
            for neuron_from in self.neuronsLP[layer_ind]:
                for neuron_to in self.neuronsLP[layer_ind + 1]:
                    Synapse(self, neuron_from, neuron_to)

    def feed_forward(self, sample):
        """ Feeds the input layer with a sample and returns the output layer values as a NumPy array """

        # Paste data into the net
        for input_neuron, x_i in zip(self.neuronsLP[0], sample):
            input_neuron.feed(x_i)

        # Activate hidden and output neurons gradually
        for layer_ind in self.l_indexes[1:]:
            for neuron in self.neuronsLP[layer_ind]:
                neuron.activate()

        return np.array([output_neuron.read() for output_neuron in self.neuronsLP[self.ol_index]], dtype=float)

    def feed_forward_fast(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def map_params(self):
        # Weights
        for synapse in self.synapsesG:
            synapse.set_weight()

        # Biases
        for neuron in self.neuronsG:
            neuron.set_bias()

    def evaluate(self, X, y, tolerance=0.1, print_all_samples=False):

        print '\n------- Evaluation of net : ' + self.name + ' (tolerance: ' + str(tolerance) + ')'

        total_err = float()
        n_correct = 0
        n_miss = 0
        for sample, target in zip(X, y):
            y_hat = self.feed_forward(sample)
            err = 0.5 * sum((target - y_hat) ** 2)
            if err <= tolerance:
                n_correct += 1
            else:
                n_miss += 1
            total_err += err
            if print_all_samples:
                print 'x:', sample, ', actual:', target, ', predict:', y_hat, 'error:', round(err, 6)

        print '\nAverage_error:', round(total_err / len(X), 6)
        print 'n_correct:', str(n_correct) + '/' + str(len(X))
        print 'n_miss:', str(n_miss) + '/' + str(len(X))
        print 'Success:', str((float(n_correct) / len(X)) * 100.0) + ' %\n--------------------------'

    def print_net(self):
        for neuron in self.neuronsG:
            if not neuron.dead:
                print '\n\n', neuron.id, neuron.activity
                try:
                    print ', synapses_out:', [syn.id for syn in neuron.synapses_out]
                except AttributeError:
                    pass
                except TypeError:
                    pass
                try:
                    print ', synapses_in:', [syn.id for syn in neuron.synapses_in]
                except AttributeError:
                    pass
                except TypeError:
                    pass

        for synapse in self.synapsesG:
            print '\n\n', synapse.id, synapse.weight

    def copy(self):
        net_copy = NeuralNet(program=self.program, name=self.name + '_copy', structure=self.structure[:])

        net_copy.weights = list()
        for arr in self.weights:
            net_copy.weights.append(np.array(arr, copy=True))

        net_copy.biases = list()
        for arr in self.biases:
            net_copy.biases.append(np.array(arr, copy=True))

        for synapse_new, synapse_old in zip(net_copy.synapsesG, self.synapsesG):
            synapse_new.init_weight = synapse_old.init_weight

        for synapse in net_copy.synapsesG[:]:
            synapse.set_weight()
            if synapse.weight == 0:
                synapse.remove_self()

        net_copy.labels = self.labels

        return net_copy

    def prepare_data(self, samples, targets):
        labels = sorted(np.unique(targets).tolist())
        results = [output_layer(position=labels.index(y_i), n_neurons=len(labels)) for y_i in targets]
        data = zip([np.reshape(x_i, (len(x_i), 1)) for x_i in samples], results)
        return data, labels

    def fit(self, X, y, X_val=None, y_val=None):
        training_data, self.labels = self.prepare_data(X, y)
        if X_val and y_val:
            validation_data, _lab = self.prepare_data(X_val, y_val)
        else:
            validation_data = None

        self.learning.train(training_data=training_data, validation_data=validation_data)

    def try_to_fit(self, X, y, X_val, y_val, req_acc):
        training_data, self.labels = self.prepare_data(X, y)
        if X_val and y_val:
            validation_data, _lab = self.prepare_data(X_val, y_val)
        else:
            validation_data = None

        return self.learning.try_to_train(training_data=training_data, validation_data=validation_data, req_acc=req_acc)

    def predict(self, samples):
        samples = [np.reshape(x_i, (len(x_i), 1)) for x_i in samples]
        return np.array([self.labels[np.argmax(self.feed_forward_fast(a=x))] for x in samples])
