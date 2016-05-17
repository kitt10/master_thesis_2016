#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.create_mnist_dataset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a MNIST dataset (after downloading it from the Yan LeCun's page.

    @arg n_samples          : number of samples per class
    @arg data_split         : training : validation : testing data split
    @arg destination_name   : dataset filename
"""

import argparse
from sys import stderr
from time import gmtime, strftime
from shelve import open as open_shelve
import gzip
import cPickle
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sknn classifier and saves it.')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('mnist_%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='Dataset filename')

    return parser.parse_args()


def load_data_wrapper(data_src):
    f = gzip.open(data_src+'/mnist.pkl.gz', 'rb')
    tr_d, va_d, te_d = cPickle.load(f)
    f.close()

    the_x = dict()
    the_y = dict()

    the_x['training'] = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    the_y['training'] = tr_d[1]
    the_x['validation'] = [np.reshape(x, (784, 1)) for x in va_d[0]]
    the_y['validation'] = va_d[1]
    the_x['testing'] = [np.reshape(x, (784, 1)) for x in te_d[0]]
    the_y['testing'] = te_d[1]
    return the_x, the_y


def vectorized_result(j, output_neurons):
    e = np.zeros((output_neurons, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    args = parse_arguments()
    destination = '../cache/datasets/mnist/'+args.destination_name+'.ds'

    ''' Loading data '''
    print '\n\n ## Loading data...'
    x, y = load_data_wrapper('../cache/downloads')
    print 'Got dataset:', len(x['training']), ':', len(x['validation']), ':', len(x['testing'])

    ''' Saving dataset '''
    print '\n\n ## Saving dataset as', destination

    dataset = open_shelve(destination, 'c')
    dataset['x'] = x
    dataset['y'] = y
    dataset['size'] = (len(x['training']), len(x['validation']), len(x['testing']))
    dataset.close()

    print 'Dataset dumped.'
