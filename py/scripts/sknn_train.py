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
    @arg destination_name   : trained classifier filename
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
    layers = [Layer('Rectifier', units=n_neurons) for n_neurons in args.structure]+[Layer('Softmax')]
    learning_rate = args.learning_rate
    n_iter = args.n_iter
    destination_name = '../cache/trained/sknn_'+args.destination_name+'.net'

    ''' Creating the neural net classifier '''
    nn_classifier = Classifier(layers=layers, learning_rate=learning_rate, n_iter=n_iter, n_stable=50, verbose=True)

    ''' Loading dataset and training '''
    print '\n\n ## Loading dataset...'
    dataset = open_shelve(dataset_dir, 'r')

    print '\n\n ## Fitting the training data...'
    nn_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))

    ''' Getting results on testing set '''
    print '\n\n ## Testing...'
    y_pred = nn_classifier.predict(np.array(dataset['x']['testing']))
    labels = nn_classifier.classes_[0].tolist()

    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    c_report = classification_report(np.array(dataset['y']['testing']), y_pred)
    cm = confusion_matrix(y_true=np.array(dataset['y']['testing']), y_pred=y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print '\n # SKNN net accuracy score on testing data:', colored(str(c_accuracy), 'green')
    print '\n # SKNN net classification report on testing data:\n', colored(str(c_report), 'cyan')
    print '\n # SKNN net confusion matrix on testing data:\n'
    print_cm(cm=cm, labels=labels)
    print '\n'
    print_cm(cm=cm_normalized, labels=labels, normed=True)
    dataset.close()

    ''' Saving trained classifier '''
    print '\n\n ## Saving trained classifier...'
    clf = open_shelve(destination_name, 'c')
    clf['classifier'] = nn_classifier
    clf['dataset'] = args.dataset
    clf['training_params'] = (args.structure, learning_rate, n_iter)
    clf['skills'] = (c_accuracy, c_report, cm)
    clf.close()
    print '\n ## SKNN net dumped as', destination_name
