#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    kitt_nn.nn_tool.nn_dataset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Desc
"""

import numpy as np
from random import choice, uniform
from math import sin, cos, pi


class Dataset(object):

    def __init__(self, dataset_name):
        self.name = dataset_name
        self.description = None
        self.training_data = None
        self.validation_data = None
        self.testing_data = None
        self.n_samples_training = None
        self.n_samples_validation = None
        self.n_samples_testing = None
        self.len_input = None
        self.len_output = None
        self.n_samples_per_class = None
        self.splitting = None

        self.train_samples = list()
        self.train_targets = list()
        self.val_samples = list()
        self.val_targets = list()
        self.test_samples = list()
        self.test_targets = list()

        # Net is assigned later, when created
        self.net = None

    def set_data(self, training_data, validation_data=None, testing_data=None):
        self.training_data = training_data
        self.n_samples_training = len(self.training_data)
        if validation_data:
            self.validation_data = validation_data
            self.n_samples_validation = len(self.validation_data)
        else:
            self.n_samples_validation = 0
        if testing_data:
            self.testing_data = testing_data
            self.n_samples_testing = len(self.testing_data)
        else:
            self.n_samples_testing = 0
        self.len_input = len(self.training_data[0][0])
        self.len_output = len(self.training_data[0][1])

    def evaluate(self, on_data):
        if on_data == 'training':
            data = self.training_data
            n_data = self.n_samples_training
        elif on_data == 'testing':
            data = self.testing_data
            n_data = self.n_samples_testing
        elif on_data == 'validation':
            data = self.validation_data
            n_data = self.n_samples_validation
        else:
            raise AttributeError
        return data, n_data


class XOR(Dataset):

    def __init__(self):
        Dataset.__init__(self, dataset_name='xor_0.01')
        self.description = 'A common XOR function.' \
                           '\n[0, 0] -> 1' \
                           '\n[1, 0] -> 0' \
                           '\n[0, 1] -> 0' \
                           '\n[1, 1] -> 1'
        self.n_samples_per_class = 10000
        self.splitting = (0.8, 0.1, 0.1)
        self.generate_data()

    def generate_data(self):
        self.train_samples = list()
        self.train_targets = list()
        self.val_samples = list()
        self.val_targets = list()
        self.test_samples = list()
        self.test_targets = list()

        for ni in range(self.n_samples_per_class):
            ''' sample for class 0 '''
            x0 = uniform(-0.5, 0.5)
            y0 = uniform(-0.49, 0.49)

            ''' sample for class 1 '''
            x1 = uniform(-0.5, 0.5)
            y1 = choice([uniform(-1.0, -0.5), uniform(0.5, 1.0)])

            ''' rotate points in space, 45deg '''
            x0_r = x0*cos(pi/4) - y0*sin(pi/4)
            y0_r = y0*cos(pi/4) + x0*sin(pi/4)
            x1_r = x1*cos(pi/4) - y1*sin(pi/4)
            y1_r = y1*cos(pi/4) + x1*sin(pi/4)

            ''' train/val/test split '''
            bounds = (self.n_samples_per_class*self.splitting[0], self.n_samples_per_class*(self.splitting[0]+self.splitting[1]))
            if ni < bounds[0]:
                self.train_samples.append([x0_r, y0_r])
                self.train_samples.append([x1_r, y1_r])
                self.train_targets.append([0.0])
                self.train_targets.append([1.0])
            elif bounds[0] <= ni < bounds[1]:
                self.val_samples.append([x0_r, y0_r])
                self.val_samples.append([x1_r, y1_r])
                self.val_targets.append([0.0])
                self.val_targets.append([1.0])
            else:
                self.test_samples.append([x0_r, y0_r])
                self.test_samples.append([x1_r, y1_r])
                self.test_targets.append([0.0])
                self.test_targets.append([1.0])

        training_data = zip([np.reshape(x, (2, 1)) for x in self.train_samples], [np.reshape(y, (1, 1)) for y in self.train_targets])
        validation_data = zip([np.reshape(x, (2, 1)) for x in self.val_samples], [np.reshape(y, (1, 1)) for y in self.val_targets])
        testing_data = zip([np.reshape(x, (2, 1)) for x in self.test_samples], [np.reshape(y, (1, 1)) for y in self.test_targets])
        self.set_data(training_data=training_data, validation_data=validation_data, testing_data=testing_data)

    def evaluate(self, clf, on_data='testing', tol=0.1):
        data, n_data = Dataset.evaluate(self, on_data=on_data)
        test_results = [(clf.predict(x), y) for (x, y) in data]
        correctly_classified = sum(int(abs(x-y) <= tol) for (x, y) in test_results)
        return float(correctly_classified)/n_data
