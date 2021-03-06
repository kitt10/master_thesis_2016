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
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from time import gmtime, strftime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from termcolor import colored
from kitt_nn.nn_structure.kitt_net import NeuralNet
from kitt_nn.nn_tool.nn_learning import BackPropagation
from kitt_nn.nn_tool.nn_function import print_cm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a kitt net and saves it.')
    parser.add_argument('-ta', '--task', type=str, default='amter', choices=['amter', 'mnist', 'xor'],
                        help='Task name')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset filename to train on')
    parser.add_argument('-s', '--structure', type=int, default=[100], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.03,
                        help='Learning rate for backpropagation')
    parser.add_argument('-i', '--n_iter', type=int, default=500,
                        help='Number of iterations (epochs)')
    parser.add_argument('-na', '--name_appendix', type=str, default='',
                        help='App. to the filename')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    dataset_dir = '../cache/datasets/'+args.task+'/'+args.dataset+'.ds'
    learning_rate = args.learning_rate
    n_iter = args.n_iter

    destination_name = args.dataset+'&'+str(learning_rate)+'_'+str(n_iter)+'_'+str(args.structure)+'_'+args.name_appendix
    destination = '../cache/trained/kitt_'+destination_name+'.net'

    ''' Loading dataset and training '''
    print '\n\n ## Loading dataset', args.dataset, '...'
    dataset = open_shelve(dataset_dir, 'r')

    net_structure = [len(dataset['x']['training'][0])]+args.structure+[len(np.unique(dataset['y']['training']))]
    net = NeuralNet(program=None, name=str(net_structure), structure=net_structure)
    net.learning = BackPropagation(program=None, net=net, learning_rate=learning_rate, n_iter=n_iter)

    print '\n\n ## Fitting the training data...'
    acc_list, err_list, time_list = net.fit(X=dataset['x']['training'], y=dataset['y']['training'],
                                            X_val=dataset['x']['validation'], y_val=dataset['y']['validation'])

    ''' Getting results on a testing set '''
    print '\n\n ## Testing...'
    y_pred = net.predict(dataset['x']['testing'])

    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    c_report = classification_report(np.array(dataset['y']['testing']), y_pred)
    cm = confusion_matrix(y_true=np.array(dataset['y']['testing']), y_pred=y_pred, labels=net.labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    dataset.close()

    ''' Saving trained classifier '''
    print '\n\n ## Saving trained classifier...'
    clf = open_shelve(destination, 'c')
    clf['net'] = (net.structure, net.weights, net.biases, net.labels)
    clf['training_params'] = (args.structure, learning_rate, n_iter)
    clf['skills'] = (c_accuracy, c_report, cm)
    clf['training_eval'] = (acc_list, err_list, time_list)
    clf.close()

    print '\n ## Kitt net dumped as', destination_name, '--------------------------------------'
    print '$ Accuracy:', colored(str(c_accuracy), 'green')
    print '\n$ Classification report:\n', colored(str(c_report), 'cyan')
    print '\n$ Confusion matrix:\n'
    print_cm(cm=cm, labels=net.labels)
    print '\n'
    print_cm(cm=cm_normalized, labels=net.labels, normed=True)
    print '------------------------------------------------------------------------------------\n\n'
