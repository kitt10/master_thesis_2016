#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.sknn_train
    ~~~~~~~~~~~~~~~~~~

    This script trains a classifier provided by sknn library.

    @arg dataset            : folder + name of the dataset file
    @arg structure          : number of layers and neurons of the net
    @arg learning_rate      : backpropagation learning rate
    @arg n_iter             : number of training epochs
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from time import gmtime, strftime
from sknn.platform import gpu32
from sknn.mlp import Classifier, Layer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from termcolor import colored
from kitt_nn.nn_tool.nn_function import print_cm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sknn classifier and saves it.')
    parser.add_argument('-ta', '--task', type=str, default='amter', choices=['amter', 'mnist', 'xor'],
                        help='Task name')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset folder and filename to train on')
    parser.add_argument('-s', '--structure', type=int, default=[100], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005,
                        help='Learning rate for backpropagation')
    parser.add_argument('-i', '--n_iter', type=int, default=500,
                        help='Number of iterations (epochs)')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    dataset_dir = '../cache/datasets/'+args.task+'/'+args.dataset+'.ds'
    layers = [Layer('Rectifier', units=n_neurons) for n_neurons in args.structure]+[Layer('Softmax')]
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    destination_name = args.dataset + '&' + str(learning_rate) + '_' + str(n_iter)+'_'+str(args.structure)
    destination = '../cache/trained/sknn_' + destination_name + '.net'

    ''' Loading dataset and training '''
    print '\n\n ## Loading dataset', args.dataset, '...'
    dataset = open_shelve(dataset_dir, 'r')

    ''' Creating the neural net classifier '''
    '''
    nn_classifier = Classifier(layers=layers, learning_rate=learning_rate, n_iter=n_iter, n_stable=20, verbose=True,
                               batch_size=10,
                               valid_set=(np.array(dataset['x']['validation']), np.array(dataset['y']['validation'])))
    '''
    nn_classifier = Classifier(layers=layers, learning_rate=learning_rate, n_iter=n_iter, n_stable=20, verbose=True,
                               batch_size=10)

    print '## Fitting the training data...'
    nn_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))

    ''' Getting results on a testing set '''
    print '\n\n ## Testing...'
    y_pred = nn_classifier.predict(np.array(dataset['x']['testing']))
    labels = nn_classifier.classes_[0].tolist()

    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    c_report = classification_report(np.array(dataset['y']['testing']), y_pred)
    cm = confusion_matrix(y_true=np.array(dataset['y']['testing']), y_pred=y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dataset.close()

    ''' Saving trained classifier '''
    print '\n\n ## Saving trained classifier...'
    clf = open_shelve(destination, 'c')
    clf['net'] = nn_classifier
    clf['training_params'] = (args.structure, learning_rate, n_iter)
    clf['skills'] = (c_accuracy, c_report, cm)
    clf.close()

    print '\n ## SKNN net dumped as', destination_name
    print '$ Accuracy:', colored(str(c_accuracy), 'green')
    print '\n$ Classification report:\n', colored(str(c_report), 'cyan')
    print '\n$ Confusion matrix:\n'
    print_cm(cm=cm, labels=labels)
    print '\n'
    print_cm(cm=cm_normalized, labels=labels, normed=True)
    print '------------------------------------------------------------------------------------\n\n'
