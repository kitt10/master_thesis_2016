#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.create_dataset
    ~~~~~~~~~~~~~~~~~~~~~~

    This script creates a .pkl dataset out of cleaned .txt files. It can also add a signal noise.

    @arg terrains       : Terrains to be added to the dataset.
    @arg terrain_noise  : Terrain noise of data to be used.
    @arg signal_noise   : Standard deviation of a signal noise to be added to the data (percentage).
    @arg sensors        : Sensors to be used.
    @arg data_split     : Training : Validation : Testing ratio (two floats)
"""

import argparse
from sys import stderr
from functions import load_params, f_range_gen


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates a .pkl dataset from cleaned .txt data.')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be involved (integers)')
    parser.add_argument('-tn', '--terrain_noise', type=str, nargs='+', default=noise_types, choices=noise_types,
                        help='Terrain noise to be used')
    parser.add_argument('-sn', '--signal_noise', type=float,
                        default=0.0, choices=list(f_range_gen(start=0.0, stop=0.1, step=0.005)),
                        help='Signal noise standard deviation (percentage)')
    parser.add_argument('-s', '--sensors', type=str, nargs='+', default=all_sensors, choices=all_sensors)
    parser.add_argument('-ds', '--data_split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        choices=list(f_range_gen(start=0.0, stop=1.0, step=0.01)),
                        help='Training : Validation : Testing data split')
    args_tmp = parser.parse_args()

    ''' Check args '''
    if abs(sum(args_tmp.data_split)-1) > 1e-5:
        stderr.write('Error: data_split args must give 1.0 together.\n')
        exit()
    else:
        return args_tmp

if __name__ == '__main__':
    terrain_types, all_sensors, noise_types = load_params('terrain_types', 'sensors', 'noise_types')
    args = parse_arguments()