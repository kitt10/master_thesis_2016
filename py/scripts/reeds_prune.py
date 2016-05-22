#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.reeds_prune
    ~~~~~~~~~~~~~~~~~~~

    This script implements a brute force pruning method based on the paper by Reeds.

    @arg net                : name of the classifier file
    @arg req_accuracy       : the required accuracy for the pruned net
    @arg learning_rate      : learning rate while pruning
    @arg max_iter           : maximum number of epochs
    @arg n_stable           : n iterations in history to find stability
    @arg destination_name   : destination_name for the pruned net
"""

import argparse
from time import strftime, gmtime
import numpy as np
from shelve import open as open_shelve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from kitt_nn.nn_structure.kitt_net import NeuralNet
from kitt_nn.nn_tool.nn_learning import BackPropagation
from kitt_nn.nn_tool.nn_function import print_cm
from termcolor import colored


def parse_arguments():
    parser = argparse.ArgumentParser(description='Classifies testing data.')
    parser.add_argument('-n', '--net', type=str, required=True,
                        help='Net filename to be pruned')
    parser.add_argument('-ds', '--dataset', type=str, required=True,
                        help='Dataset filename to train on')
    parser.add_argument('-ra', '--req_accuracy', type=float, default=0.99,
                        help='Required accuracy for the pruned net')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.03,
                        help='Learning rate for backpropagation')
    parser.add_argument('-mi', '--max_iter', type=int, default=20,
                        help='Maximum number of iterations (epochs)')
    parser.add_argument('-ns', '--n_stable', type=int, default=10,
                        help='N iterations to fire a stability')
    parser.add_argument('-na', '--name_appendix', type=str, default='',
                        help='App. to the filename')
    return parser.parse_args()


def report(y_pred, y_true, labels):
    c_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    c_report = classification_report(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    cm_normed = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print '\n # -------------------------------------------------------'
    print '\n # Net accuracy score:', colored(str(c_accuracy), 'green')
    print '\n # Net classification report:\n', colored(str(c_report), 'cyan')
    print '\n # Net confusion matrix:\n'
    print_cm(cm=cm, labels=labels)
    print '\n'
    print_cm(cm=cm_normed, labels=labels, normed=True)
    print '\n # ------------------------------------------------------- \n'

    return c_accuracy, c_report, cm


def cut_synapses(net, level):
    """ cuts synapses """
    if level > 0:
        th_change = np.percentile([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG], level)
    else:
        th_change = min([abs(synapse.get_weight()-synapse.init_weight) for synapse in net.synapsesG])

    c = 0
    for synapse in net.synapsesG[:]:
        synapse.set_weight()
        if abs(synapse.weight-synapse.init_weight) <= th_change:
            synapse.remove_self()
            c += 1
    n_to_remove = c
    print ' Trying to remove', n_to_remove, 'synapses. May I? Percentile:', level
    return net, n_to_remove


def able_to_learn(tested_net):
    tested_net.learning = BackPropagation(program=None, net=tested_net, learning_rate=learning_rate, n_iter=max_epochs,
                                          n_stable=n_stable)
    return tested_net.try_to_fit(X=dataset['x']['training'], y=dataset['y']['training'],
                                 X_val=dataset['x']['validation'], y_val=dataset['y']['validation'],
                                 req_acc=req_accuaracy)


if __name__ == '__main__':

    args = parse_arguments()
    net_dir = '../cache/trained/'+args.net+'.net'
    destination = '../cache/pruned/'+args.net+'_reed_p_'+args.name_appendix+'.net'
    learning_rate = args.learning_rate
    max_epochs = args.max_iter
    n_stable = args.n_stable
    dataset_dir = args.dataset

    ''' Loading the classifier and testing data '''
    net_file = open_shelve(net_dir, 'c')
    nn_classifier = net_file['net']
    structure = nn_classifier[0]
    weights = nn_classifier[1]
    biases = nn_classifier[2]
    labels = nn_classifier[3]
    print '\n\n ##Net loaded. Training parameters:', net_file['training_params']
    net_file.close()

    net = NeuralNet(program=None, name=str(structure), structure=structure)
    net.weights = weights
    net.biases = biases
    net.labels = labels
    net.map_params()

    dataset = open_shelve('../cache/datasets/'+dataset_dir+'.ds', 'c')
    print 'Dataset loaded.'

    ''' Initial classification reports '''
    print '\n\n ## TRAINING DATA initial prediction...'
    report(y_pred=net.predict(dataset['x']['training']), y_true=np.array(dataset['y']['training']),
                 labels=net.labels)

    print '\n\n ## VALIDATION DATA initial prediction...'
    req_accuaracy = report(y_pred=net.predict(dataset['x']['validation']), y_true=np.array(dataset['y']['validation']),
                           labels=net.labels)[0]
    #if args.req_accuracy < req_accuaracy:
    req_accuaracy = args.req_accuracy
    print '\n '+colored('=> @required_accuracy has been set to '+str(req_accuaracy), 'blue')

    ''' Pruning '''
    pruning_step = 0
    pruning_done = False
    s_i = 0

    # stats containers
    acc_list = list()
    n_syn_list = list()
    structure_list = list()

    print '\n\n ## PRUNING HAS STARTED...'
    while True:
        pruning_step += 1

        ''' Make a copy of the net and remove one synapse '''
        net_tmp = net.copy()
        try:
            print '\n# Pruning step', pruning_step, 'removed:', net_tmp.synapsesG[s_i].id
            net_tmp.synapsesG[s_i].set_weight()
            net_tmp.synapsesG[s_i].remove_self()
        except IndexError:
            break

        ''' Save stats '''
        acc_list.append(accuracy_score(y_pred=net.predict(dataset['x']['testing']),
                                       y_true=np.array(dataset['y']['testing'])))
        n_syn_list.append(len(net.synapsesG))
        structure_list.append([sum([not neuron.dead for neuron in layer]) for layer in net.neuronsLP.values()])

        if able_to_learn(tested_net=net_tmp):
            print 'Accepted,', acc_list[-1], structure_list[-1], len(net_tmp.synapsesG)
            net = net_tmp.copy()
        else:
            s_i += 1

    print 'Final structure:',
    final_structure_neurons = list()
    for layer in net.neuronsLP.values():
        final_structure_neurons.append([neuron.dead for neuron in layer])
        print sum([not neuron.dead for neuron in layer]),

    print '\n\n ## TESTING DATA final structure prediction...'
    c_accuracy, c_report, cm = report(y_pred=net.predict(dataset['x']['testing']),
                                       y_true=np.array(dataset['y']['testing']), labels=net.labels)

    dataset.close()

    ''' Saving trained classifier '''
    print '\n\n ## Saving pruned classifier to', destination, '...'
    clf = open_shelve(destination, 'c')
    clf['net'] = (net.structure, net.weights, net.biases, net.labels, net.synapses_exist, final_structure_neurons)
    clf['training_params'] = ([sum([not neuron.dead for neuron in layer]) for layer in net.neuronsLP.values()], learning_rate, max_epochs, n_stable)
    clf['skills'] = (c_accuracy, c_report, cm)
    clf['pruning_eval'] = (acc_list, n_syn_list, structure_list)
    clf.close()