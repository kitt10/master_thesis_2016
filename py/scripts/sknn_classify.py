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
from sklearn.metrics import classification_report


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
    dataset_dir = clf['dataset']
    print '\n\n ## Classification : training parameters:', clf['training_params']
    clf.close()

    dataset = open_shelve('../cache/datasets/amos_terrains_sim/'+dataset_dir+'.ds', 'c')
    ''' Classifying '''
    print 'NN classification report on validation data:\n%s\n' % \
          classification_report(np.array(dataset['y']['testing']),
                                nn_classifier.predict(np.array(dataset['x']['testing'])))
    dataset.close()
