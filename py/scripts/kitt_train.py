#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.kitt_train
    ~~~~~~~~~~~~~~~~~~

    This script trains a neural net developed by kitt :-).

    @arg dataset            : folder + name of the dataset file
    @arg structure          : number of layers and neurons of the net
    @arg learning_rate      : backpropagation learning rate
    @arg n_iter             : number of training epochs
    @arg destination_name   : trained classifier filename
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from time import gmtime, strftime
from sklearn.metrics import classification_report
from kitt_nn.nn_structure.kitt_net import NeuralNet
from kitt_nn.nn_tool.nn_learning import BackPropagation


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sknn classifier and saves it.')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset folder and filename to train on')
    parser.add_argument('-s', '--structure', type=int, default=[100], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005,
                        help='Learning rate for backpropagation')
    parser.add_argument('-i', '--n_iter', type=int, default=500,
                        help='Number of iterations (epochs)')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='Trained classifier filename')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    dataset_dir = '../cache/datasets/'+args.dataset+'.ds'
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    destination_name = '../cache/trained/kitt_'+args.destination_name+'.net'

    ''' Loading dataset and training '''
    dataset = open_shelve(dataset_dir, 'r')

    net_structure = [len(dataset['x']['training'][0])]+args.structure+[len(np.unique(dataset['y']['training']))]
    net = NeuralNet(program=None, name=str(net_structure), structure=net_structure)
    net.learning = BackPropagation(program=None, net=net, learning_rate=learning_rate, n_iter=n_iter)

    net.fit(X=dataset['x']['training'], y=dataset['y']['training'],
            X_val=dataset['x']['validation'], y_val=dataset['y']['validation'])
    dataset.close()
    exit()

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
