#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.sknn_train
    ~~~~~~~~~~~~~~~~~~

    This script trains a classifier provided by sknn library.

    @arg dataset            : name of the dataset file
    @arg structure          : number of layers and neurons of the net
    @arg learning_rate      : backpropagation learning rate
    @arg n_iter             : number of training epochs
    @arg destination_name   : trained classifier filename
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from time import gmtime, strftime
from sknn.platform import gpu32
from sknn.mlp import Classifier, Layer
from sklearn.metrics import classification_report


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sknn classifier and saves it.')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset name to train on')
    parser.add_argument('-s', '--structure', type=int, default=[100], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005,
                        help='Learning rate for backpropagation')
    parser.add_argument('-i', '--n_iter', type=int, default=500,
                        help='Number of iterations (epochs)')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('sknn_%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='Trained classifier filename')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    dataset_dir = '../cache/datasets/amos_terrains_sim/'+args.dataset+'.ds'
    layers = [Layer('Rectifier', units=n_neurons) for n_neurons in args.structure]+[Layer('Softmax')]
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    destination_name = '../cache/trained/'+args.destination_name+'.net'

    ''' Creating the neural net classifier '''
    nn_classifier = Classifier(layers=layers, learning_rate=learning_rate, n_iter=n_iter, verbose=True)

    ''' Loading dataset and training '''
    dataset = open_shelve(dataset_dir, 'r')
    nn_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))

    print 'NN classification report on validation data:\n%s\n' % \
          classification_report(np.array(dataset['y']['validation']),
                                nn_classifier.predict(np.array(dataset['x']['validation'])))
    dataset.close()

    ''' Saving trained classifier '''
    clf = open_shelve(destination_name, 'c')
    clf['classifier'] = nn_classifier
    clf['dataset'] = args.dataset
    clf['training_params'] = (args.structure, learning_rate, n_iter)
    clf.close()
