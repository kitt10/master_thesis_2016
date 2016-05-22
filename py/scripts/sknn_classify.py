#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.sknn_classify
    ~~~~~~~~~~~~~~~~~~~~~

    This script classifies testing data with a trained classifier provided by sknn library.

    @arg clf            : name of the classifier file
"""

import argparse
import numpy as np
from shelve import open as open_shelve
from sknn.platform import gpu32
from sklearn.metrics import classification_report, accuracy_score
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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
    dataset_dir = args.dataset
    print '\n\n ## Classification : training parameters:', clf['training_params']
    clf.close()
    dataset = open_shelve('../cache/datasets/' + dataset_dir + '.ds', 'c')

    val_set = (np.reshape(np.array(dataset['x']['validation']),
                          (len(dataset['x']['validation']), len(dataset['x']['validation'][0]))),
               np.array(dataset['y']['validation']))
    nn_classifier = Classifier(layers=[Layer('Sigmoid', units=20), Layer('Softmax')],
                                               learning_rate=0.1, n_iter=500, n_stable=20, verbose=True,
                               batch_size=10, valid_set=val_set)

    rf_classifier = RandomForestClassifier(n_estimators=10)
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    svm_classifier = SVC()

    print '## Fitting the training data...'
    nn_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))
    rf_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))
    knn_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))
    svm_classifier.fit(np.array(dataset['x']['training']), np.array(dataset['y']['training']))

    ''' Classifying '''
    y_pred = nn_classifier.predict(np.array(dataset['x']['testing']))
    print 'NN classification report on testing data:\n%s\n' % \
          classification_report(np.array(dataset['y']['testing']), y_pred=y_pred)
    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    print 'Accuracy:', c_accuracy

    y_pred = rf_classifier.predict(np.array(dataset['x']['testing']))
    print '\n\nRF classification report on testing data:\n%s\n' % \
          classification_report(np.array(dataset['y']['testing']), y_pred=y_pred)
    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    print 'Accuracy:', c_accuracy

    y_pred = svm_classifier.predict(np.array(dataset['x']['testing']))
    print '\n\nSVM classification report on testing data:\n%s\n' % \
          classification_report(np.array(dataset['y']['testing']), y_pred=y_pred)
    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    print 'Accuracy:', c_accuracy

    y_pred = knn_classifier.predict(np.array(dataset['x']['testing']))
    print '\n\nK-NN classification report on testing data:\n%s\n' % \
          classification_report(np.array(dataset['y']['testing']), y_pred=y_pred)
    c_accuracy = accuracy_score(y_true=np.array(dataset['y']['testing']), y_pred=y_pred)
    print 'Accuracy:', c_accuracy
    dataset.close()
