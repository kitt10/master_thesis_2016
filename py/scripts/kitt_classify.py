#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.kitt_classify
    ~~~~~~~~~~~~~~~~~~~~~

    This script classifies testing data with a trained classifier provided by kitt :-).

    @arg clf            : name of the classifier file
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from kitt_nn.nn_structure.kitt_net import NeuralNet
from kitt_nn.nn_tool.nn_function import print_cm
from termcolor import colored


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classifies testing data.')
    parser.add_argument('-c', '--clf', type=str, required=True,
                        help='Classifier filename to classify with')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    clf_dir = '../cache/trained/'+args.clf+'.net'

    ''' Loading the classifier and testing data '''
    clf = open_shelve(clf_dir, 'c')
    nn_classifier = clf['classifier']
    structure = nn_classifier[0]
    weights = nn_classifier[1]
    biases = nn_classifier[2]
    labels = nn_classifier[3]
    dataset_dir = clf['dataset']
    print '\n\n ## Classification : training parameters:', clf['training_params']
    clf.close()

    net = NeuralNet(program=None, name=str(structure), structure=structure)
    net.weights = weights
    net.biases = biases
    net.labels = labels
    net.map_params()

    dataset = open_shelve('../cache/datasets/'+dataset_dir+'.ds', 'c')
    ''' Classifying '''
    ''' Getting results on testing set '''
    print '\n\n ## Testing...'
    y_pred = net.predict(dataset['x']['testing'])

    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    c_report = classification_report(np.array(dataset['y']['testing']), y_pred)
    cm = confusion_matrix(y_true=np.array(dataset['y']['testing']), y_pred=y_pred, labels=net.labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print '\n # Kitt net accuracy score on testing data:', colored(str(c_accuracy), 'green')
    print '\n # Kitt net classification report on testing data:\n', colored(str(c_report), 'cyan')
    print '\n # Kitt net confusion matrix on testing data:\n'
    print_cm(cm=cm, labels=net.labels)
    print '\n'
    print_cm(cm=cm_normalized, labels=net.labels, normed=True)
    dataset.close()
