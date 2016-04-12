#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.create_xor_dataset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a XOR dataset.

    @arg level              : distance between the classes
    @arg n_samples          : number of samples per class
    @arg data_split         : training : validation : testing data split
    @arg destination_name   : dataset filename
"""

import argparse
from sys import stderr
from time import gmtime, strftime
from random import choice, uniform
from math import sin, cos, pi
from numpy import reshape
from shelve import open as open_shelve
from functions import f_range_gen


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sknn classifier and saves it.')
    parser.add_argument('-l', '--level', type=float, default=0.01,
                        help='Distance between classes')
    parser.add_argument('-ns', '--n_samples', type=int, default=1000,
                        help='Number of samples per class')
    parser.add_argument('-ds', '--data_split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        choices=list(f_range_gen(start=0.0, stop=1.0, step=0.01)),
                        help='Training : Validation : Testing data split')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('xor_%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='Dataset filename')

    args_tmp = parser.parse_args()

    ''' Check args '''
    if abs(sum(args_tmp.data_split) - 1) > 1e-5:
        stderr.write('Error: data_split args must give 1.0 together.\n')
        exit()
    else:
        return args_tmp


if __name__ == '__main__':
    args = parse_arguments()

    level = args.level
    n_samples = args.n_samples
    split_bounds = (n_samples*args.data_split[0], n_samples*(args.data_split[0]+args.data_split[1]))
    destination = '../cache/datasets/xor/'+args.destination_name+'.ds'

    ''' Generating and splitting data '''
    print '\n\n ## Generating and splitting data...'

    x = {'training': list(), 'validation': list(), 'testing': list()}
    y = {'training': list(), 'validation': list(), 'testing': list()}

    for ni in range(n_samples):
        ''' sample for class 0 '''
        x0 = uniform(-0.5, 0.5)
        y0 = uniform(-0.49, 0.49)

        ''' sample for class 1 '''
        x1 = uniform(-0.5, 0.5)
        y1 = choice([uniform(-1.0, -0.5), uniform(0.5, 1.0)])

        ''' rotate points in space, 45deg '''
        x0_r = x0 * cos(pi / 4) - y0 * sin(pi / 4)
        y0_r = y0 * cos(pi / 4) + x0 * sin(pi / 4)
        x1_r = x1 * cos(pi / 4) - y1 * sin(pi / 4)
        y1_r = y1 * cos(pi / 4) + x1 * sin(pi / 4)

        ''' train/val/test split '''
        if ni < split_bounds[0]:
            x['training'].append([x0_r, y0_r])
            x['training'].append([x1_r, y1_r])
            y['training'].append(0.0)
            y['training'].append(1.0)
        elif split_bounds[0] <= ni < split_bounds[1]:
            x['validation'].append([x0_r, y0_r])
            x['validation'].append([x1_r, y1_r])
            y['validation'].append(0.0)
            y['validation'].append(1.0)
        else:
            x['testing'].append([x0_r, y0_r])
            x['testing'].append([x1_r, y1_r])
            y['testing'].append(0.0)
            y['testing'].append(1.0)

    print 'Got dataset:', len(x['training']), ':', len(x['validation']), ':', len(x['testing'])

    ''' Saving dataset '''
    print '\n\n ## Saving dataset as', destination

    dataset = open_shelve(destination, 'c')
    dataset['x'] = x
    dataset['y'] = y
    dataset['level'] = level
    dataset['n_samples'] = n_samples
    dataset['data_split'] = args.data_split
    dataset['size'] = (len(x['training']), len(x['validation']), len(x['testing']))
    dataset.close()

    print 'Dataset dumped.'
