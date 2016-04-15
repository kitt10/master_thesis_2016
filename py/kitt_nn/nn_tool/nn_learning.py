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
from termcolor import colored


class ANNLearning(object):

    def __init__(self, program, net, learning_name):
        self.program = program
        self.name = learning_name
        self.net = net

    def train(self, *args):
        raise NotImplementedError('Learning process not defined.')


class BackPropagation(ANNLearning):

    def __init__(self, program, net, learning_rate=0.03, n_iter=100, n_stable=10, batch_size=10, verbose=True):
        ANNLearning.__init__(self, program, net, learning_name='BackPropagation')
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_stable = n_stable
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_rate = None
        self.train_error = None
        self.min_train_error = 1.0
        self.val_rate = None
        self.val_error = None
        self.min_val_error = 1.0

    def train(self, training_data, validation_data=None):

        if self.verbose:
            print '\n\n ## The training has started...'
            print ' Epoch\tOn Training Data\tOn Validation Data\tEpoch Time'
            print '--------------------------------------------------------------------'
        for i_epoch in xrange(1, self.n_iter+1):
            shuffle(training_data)
            mini_batches = [training_data[k:k+self.batch_size] for k in xrange(0, len(training_data), self.batch_size)]

            t0 = time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            epoch_time = time()-t0

            self.train_rate, self.train_error = self.evaluate(data=training_data)
            if validation_data:
                self.val_rate, self.val_error = self.evaluate(data=validation_data)

            if self.verbose:
                line = ' '+str(i_epoch)+'\t'+colored(str(format(self.train_rate, '.2f')), 'green')
                if self.train_error < self.min_train_error:
                    self.min_train_error = self.train_error
                    col = 'red'
                else:
                    col = 'magenta'
                line += '/'+colored(str(format(self.train_error, '.4f')), col)
                line += colored('\t \t'+str(format(self.val_rate, '.2f')), 'green')
                if self.val_error < self.min_val_error:
                    self.min_val_error = self.val_error
                    col = 'red'
                else:
                    col = 'magenta'
                line += '/'+colored(str(format(self.val_error, '.4f')), col)
                line += '\t \t'+colored(str(format(epoch_time, '.4f'))+' s', 'cyan')
                print line

    def try_to_train(self, training_data, validation_data, req_acc):
        err_history = list()
        err_min = 1.0

        for epoch in xrange(1, self.n_iter+1):
            net_structure = list()
            for layer in self.net.neuronsLP.values():
                net_structure.append(sum([not neuron.dead for neuron in layer]))
            shuffle(training_data)
            batches = [training_data[k:k + self.batch_size] for k in xrange(0, len(training_data), self.batch_size)]

            for mini_batch in batches:
                self.update_mini_batch(mini_batch)

            res = self.evaluate(data=validation_data)
            err_history.append(res[1])
            if err_history[-1] < err_min:
                col = 'red'
                err_min = err_history[-1]
            else:
                col = 'magenta'
            print ' # '+str(epoch).zfill(4)+' -> ',
            print colored(format(res[0], '.3f'), 'green')+'/'+colored(format(res[1], '.6f'), col),
            print ' in net: '+str(net_structure)+' with '+str(len(self.net.synapsesG))+' synapses'

            if res[0] >= req_acc:
                return True
            else:
                try:
                    for i in range(2, self.n_stable+1):
                        if err_history[-1] < err_history[-i]:
                            break
                    else:
                        return False
                except IndexError:
                    pass

    def evaluate(self, data):
        test_results = [(self.net.labels[np.argmax(self.net.feed_forward_fast(a=x))], self.net.labels[np.argmax(y)]) for (x, y) in data]
        correctly_classified = sum([int(x == y) for (x, y) in test_results])
        err_sum = 0.0
        for x, y in data:
            cl_e = [abs(cl_x[0]-cl_y[0]) for (cl_x, cl_y) in zip(self.net.feed_forward_fast(x), y)]
            err_sum += float(sum(cl_e))/len(data[0][1])
        return float(correctly_classified) / len(data), float(err_sum) / len(data)

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
