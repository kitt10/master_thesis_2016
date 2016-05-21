#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
from os import path


def load_datasets():
    dataset_paths = glob(path.join('../cache/datasets/amter/', '*.ds'))

    old_values = ('nn', 'n1p', 'n3p', 'n5p', 'n10p', 'n20p')
    new_values = ('0.00', '0.01', '0.03', '0.05', '0.10', '0.20')
    for i_p, the_path in enumerate(dataset_paths):
        for old_value, new_value in zip(old_values, new_values):
            dataset_paths[i_p] = dataset_paths[i_p].replace(old_value, new_value)

    ds_names = [the_path.split('/')[-1] for the_path in dataset_paths]
    ds_names = sorted(ds_names, key=lambda x: x[6:10])
    return ds_names


def get_ds_id(x):
    dataset_names = load_datasets()
    db = dict()
    for d_i, ds_name in enumerate(dataset_names):
        p = ds_name.split('_')
        db[p[1]+';0.'+p[2][1:]+';'+p[3]+';'+p[4]] = d_i+1
    return db[x[0]+';'+x[1]+';'+x[2]+';'+x[3]]



def print_datasets():
    print 'Found', len(dataset_names), 'datasets.'
    map_s = {'alls': 'all', 'angle': 'proprioceptive', 'foot': 'tactile'}
    map_ss = {'alls': 'a', 'angle': 'p', 'foot': 't'}
    print 'name\t', 'tn\t', 'sn\t', 'ts\t', 'sen\t', 'ter\t', 'ns'
    for d_i, ds_name in enumerate(dataset_names):
        p = ds_name.split('_')
        print 'ds_'+str(d_i+1).zfill(2)+'\t', p[1]+'\t', '0.'+p[2][1:]+'\t', p[3]+'\t', p[4]+'\t', p[5]+'\t', p[6][:-3]+'\t'

    print 'For LaTeX:'
    print 'name\t', 'tn\t', 'sn\t', 'ts\t', 'sen'
    for d_i, ds_name in enumerate(dataset_names):
        p = ds_name.split('_')
        p[2] = '0.'+p[2][1:]
        if len(p[2]) < 4:
            p[2] += '0'
        print p[1][2:]+'\_'+p[2][2:]+'\_'+p[3].zfill(2)+'\_'+map_ss[p[4]] + '\t&', p[1] + '\t&', p[2] + '\t&', p[3].zfill(2) + '\t&', map_s[p[4]] + '\t \\\\ \hline'


if __name__ == '__main__':
    dataset_names = load_datasets()
    print_datasets()