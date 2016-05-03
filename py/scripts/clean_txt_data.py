#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.clean_txt_data
    ~~~~~~~~~~~~~~~~~~~~~~

    This script checks the generated .txt files and removes bad ones.

    @arg terrains       : Terrains to be checked.
    @arg noises         : Terrain noises to be checked.
    @arg sample_len     : Minimum length of one sample (simulation steps)
"""

import argparse
import json
from glob import glob
from os import path, remove
from functions import load_params


def parse_arguments():
    parser = argparse.ArgumentParser(description='Checks generated .txt files.')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be checked (integers)')
    parser.add_argument('-n', '--noises', type=str, nargs='+', default=noise_types, choices=noise_types,
                        help='Terrain noises to be checked')
    parser.add_argument('-sl', '--sample_len', type=int, default=95,
                        help='Minimum sample length.')
    return parser.parse_args()


if __name__ == '__main__':
    terrain_types, noise_types, noise_params = load_params('terrain_types', 'noise_types', 'noise_params')
    args = parse_arguments()

    noises_to_check = args.noises
    terrains_to_check = [terrain_types[str(i)] for i in sorted(args.terrains)]
    min_sample_len = args.sample_len

    for noise_type in noises_to_check:
        for terrain_type in terrains_to_check:
            txt_samples = glob(path.join('../../data/'+noise_type+'/'+noise_params[noise_type][0]+terrain_type, '*.txt'))
            n_good_samples = 0
            for path_and_filename in sorted(txt_samples):
                with open(path_and_filename, 'r') as data_file:
                    file_str = data_file.read()

                ''' Remove the old file '''
                remove(path_and_filename)

                ''' First, check if the file is not messed '''
                tmp_str = ''
                n_steps = 0
                for line in file_str.split('\n'):
                    try:
                        if int(line.split(';')[0]) < 100:
                            tmp_str += line+'\n'
                            n_steps += 1
                    except ValueError:
                        pass

                ''' Then check the sample length '''
                if n_steps >= min_sample_len:

                    ''' Remember to re-index'''
                    n_good_samples += 1
                    ind = path_and_filename.find('job') + 3
                    new_path_and_filename = path_and_filename[:ind]+str(n_good_samples).zfill(5)+'.txt'
                    with open(new_path_and_filename, 'w') as data_file:
                        data_file.write(tmp_str)

            print 'Kept', n_good_samples, 'samples for', noise_type, 'on terrain', terrain_type
