#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.kitt_classify
    ~~~~~~~~~~~~~~~~~~~~~

    This script classifies testing data with a trained classifier provided by kitt :-).

    @arg clf            : name of the classifier file
"""

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 18
import argparse
import numpy as np
from shelve import open as open_shelve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from kitt_nn.nn_structure.kitt_net import NeuralNet
from kitt_nn.nn_tool.nn_function import print_cm
from functions import load_params
from termcolor import colored


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classifies testing data.')
    parser.add_argument('-c', '--clf', type=str, required=True,
                        help='Classifier filename to classify with')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset to classify on')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    clf_dir = '../cache/trained/'+args.clf+'.net'

    ''' Loading the classifier and testing data '''
    clf = open_shelve(clf_dir, 'c')
    nn_classifier = clf['net']
    structure = nn_classifier[0]
    weights = nn_classifier[1]
    biases = nn_classifier[2]
    labels = nn_classifier[3]
    dataset_dir = args.dataset
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

    terrain_ids = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
    terrain_types = load_params('terrain_types')[0]
    terrains = [terrain_types[str(t_id)] for t_id in terrain_ids]
    plt.matshow(cm_normalized, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(14), terrains, rotation=45)
    plt.yticks(range(14), terrains)
    for t1_i, terrain1 in enumerate(terrains):
        for t2_i, terrain2 in enumerate(terrains):
            if cm_normalized[t1_i][t2_i] >= 0.01:
                plt.text(t2_i, t1_i, round(cm_normalized[t1_i][t2_i], 2), va='center', ha='center', fontsize=12)
    plt.show()
    #plt.savefig('../../thesis/img/amter_classification_nn_cm.eps', bbox_inches='tight', pad_inches=0.1)
    dataset.close()
