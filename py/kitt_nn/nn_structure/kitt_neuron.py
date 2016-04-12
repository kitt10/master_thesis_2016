#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_structure.kitt_neuron
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Desc
"""

from kitt_nn.nn_tool.nn_function import sigmoid


class Neuron(object):

    def __init__(self, net, layer_ind, layer_id):
        self.net = net
        self.layer_ind = layer_ind
        self.activity = float()
        self.synapses_in = None
        self.synapses_out = None
        self.z = None                   # Unactivated value of neuron (sometimes also 'a')
        self.d = None                   # Delta : for back-propagation
        self.bias = float()
        self.dead = False

        # Register self
        self.gind = len(self.net.neuronsG)
        self.net.neuronsG.append(self)
        self.layer_pos = len(self.net.neuronsLP[self.layer_ind])
        self.net.neuronsLP[self.layer_ind].append(self)
        self.id = layer_id+'.'+str(layer_ind)+'.'+str(self.layer_pos)
        self.net.synapsesNN[self] = dict()

        # Graphics
        self.g_body = None
        self.g_axon = None
        self.g_x = None
        self.g_y = None
        self.g_axon_x = None
        self.g_axon_y = None

    def activate(self):
        self.z = sum([synapse.neuron_from.activity*synapse.weight for synapse in self.synapses_in]) + self.bias
        self.activity = sigmoid(self.z)

    def get_bias(self):
        return self.net.biases[self.layer_ind+1][self.layer_pos][0]

    def set_bias(self):
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]

    def set_bias_b(self, b):
        self.net.biases[self.layer_ind-1][self.layer_pos][0] = b
        self.set_bias()

    def set_dead(self):
        self.dead = True
        try:
            for synapse_in in self.synapses_in[:]:
                synapse_in.remove_self()
        except TypeError:
            pass

        try:
            for synapse_out in self.synapses_out[:]:
                synapse_out.remove_self()
        except TypeError:
            pass


class InputNeuron(Neuron):

    def __init__(self, net, layer_ind):
        Neuron.__init__(self, net, layer_ind, 'i')
        self.synapses_out = list()

    def activate(self):
        pass

    def feed(self, x):
        self.activity = x

    def get_bias(self):
        return None

    def set_bias(self):
        pass

    def set_bias_b(self, b):
        self.bias = b


class HiddenNeuron(Neuron):

    def __init__(self, net, layer_ind):
        Neuron.__init__(self, net, layer_ind, 'h')
        self.synapses_in = list()
        self.synapses_out = list()
        self.z = float()
        self.d = float()
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]


class OutputNeuron(Neuron):

    def __init__(self, net, layer_ind):
        Neuron.__init__(self, net, layer_ind, 'o')
        self.synapses_in = list()
        self.z = float()
        self.d = float()
        self.bias = self.net.biases[self.layer_ind-1][self.layer_pos][0]

    def read(self):
        return self.activity
