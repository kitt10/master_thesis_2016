#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_tool.nn_learning
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Desc
"""


import numpy as np
from random import shuffle
from time import time
from nn_function import sigmoid, sigmoid_prime


class ANNLearning(object):

    def __init__(self, program, net, learning_name):
        self.program = program
        self.name = learning_name
        self.net = net

    def train(self, *args):
        raise NotImplementedError('Learning process not defined.')


class BackPropagation(ANNLearning):

    def __init__(self, program, net, learning_rate=0.005, n_iter=100, batch_size=10, verbose=True):
        ANNLearning.__init__(self, program, net, learning_name='BackPropagation')
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_rate = None
        self.val_rate = None

    def train(self, training_data, validation_data=None):

        if self.verbose:
            print '\n\n ## Learning started...'
            print ' Epoch\tOn Training Data\tOn Validation Data\tEpoch Time'
            print '--------------------------------------------------------------------'
        for i_epoch in xrange(1, self.n_iter+1):
            shuffle(training_data)
            mini_batches = [training_data[k:k+self.batch_size] for k in xrange(0, len(training_data), self.batch_size)]

            t0 = time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            epoch_time = time()-t0

            self.train_rate = self.evaluate(data=training_data)
            if validation_data:
                self.val_rate = self.evaluate(data=validation_data)

            if self.verbose:
                print ' '+str(i_epoch)+'\t \t'+str(format(self.train_rate, '.4f'))+'\t \t \t'+str(format(self.val_rate, '.4f'))+\
                      '\t \t'+str(format(epoch_time, '.4f'))+' s'

    def evaluate(self, data):
        test_results = [(np.argmax(self.net.feed_forward_fast(x)), np.argmax(y)) for (x, y) in data]
        correctly_classified = sum(int(x == y) for (x, y) in test_results)
        return float(correctly_classified) / len(data)

    def update_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.do_backpropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.net.weights = np.multiply([w-(self.learning_rate/len(mini_batch))*nw for w, nw in zip(self.net.weights, nabla_w)], self.net.synapses_exist)
        self.net.biases = [b-(self.learning_rate/len(mini_batch))*nb for b, nb in zip(self.net.biases, nabla_b)]

    def do_backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.net.biases]
        nabla_w = [np.zeros(w.shape) for w in self.net.weights]

        # feedforward
        activation = x
        activations = [x]       # list to store all the activations, layer by layer
        zs = list()             # list to store all the z vectors, layer by layer
        for b, w in zip(self.net.biases, self.net.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, len(self.net.structure)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.net.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return nabla_b, nabla_w
