#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_structure.kitt_synapse
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Desc
"""


class Synapse(object):

    def __init__(self, net, neuron_from, neuron_to):
        self.net = net
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]
        self.init_weight = self.weight

        # Register self
        self.gind = len(self.net.synapsesG)
        self.net.synapsesG.append(self)
        self.net.synapsesNN[neuron_from][neuron_to] = self
        self.neuron_from.synapses_out.append(self)
        self.neuron_to.synapses_in.append(self)
        self.id = neuron_from.id+'->'+neuron_to.id
        self.net.synapses_exist[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = 1.0

        # Graphics
        self.g_gray_value = None
        self.g_line = None

    def get_weight(self):
        return self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]

    def set_weight(self):
        self.weight = self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos]

    def set_weight_w(self, w):
        self.net.weights[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = w
        self.set_weight()

    def remove_self(self):
        self.net.synapses_exist[self.neuron_from.layer_ind][self.neuron_to.layer_pos][self.neuron_from.layer_pos] = 0.0
        self.set_weight_w(0.0)
        try:
            self.neuron_to.synapses_in.remove(self)
            self.neuron_from.synapses_out.remove(self)
            self.net.synapsesG.remove(self)
        except ValueError:
            pass

        if not self.neuron_from.synapses_out:
            self.neuron_from.set_dead()
        if not self.neuron_to.synapses_in:
            self.neuron_to.set_dead()
        try:
            del self.net.synapsesNN[self.neuron_from][self.neuron_to]
        except KeyError:
            pass

        del self