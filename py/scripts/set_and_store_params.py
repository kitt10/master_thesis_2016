#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.set_and_store_params
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script sets some global parameters of the whole project and stores them as .json files.

    @arg params       : Parameters to be reset and restored.
"""

import argparse
import json

__author__ = 'Martin Bulin'
__copyright__ = 'Copyright 2016, Master Thesis'
__credits__ = ['Martin Bulin', 'Tomas Kulvicius', 'Poramate Manoonpong']
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Martin Bulin'
__email__ = 'bulinmartin@gmail.com'
__status__ = 'Development'

parser = argparse.ArgumentParser(description='Sets global project parameters and stores them as .json files.')
parser.add_argument('-p', '--params', type=str, nargs='+',
                    default=['terrain_types', 'terrain_qualities', 'qualities_ranges', 'env', 'noise_types'],
                    choices=['terrain_types', 'terrain_qualities', 'qualities_ranges', 'env', 'noise_types'],
                    help='Params to be reset and restored')
args = parser.parse_args()

terrain_types = {1: 'concrete', 2: 'mud', 3: 'ice', 4: 'sand', 5: 'gravel', 6: 'grass', 7: 'swamp', 8: 'rock',
                 9: 'tiles', 10: 'snow', 11: 'rubber', 12: 'carpet', 13: 'wood', 14: 'plastic', 15: 'foam'}

terrain_qualities = ('roughness', 'slipperiness', 'hardness', 'elasticity', 'height')
qualities_ranges = {'roughness': (0.0, 10.0), 'slipperiness': (0.0, 100.0), 'hardness': (0.0, 100.0),
                    'elasticity': (0.0, 2.0), 'height': (0.0, 0.1)}

env = {'concrete':  {'roughness': 10.0, 'slipperiness': 0.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0},
       'mud':       {'roughness': 0.5, 'slipperiness': 5.0, 'hardness': 0.5, 'elasticity': 0.5, 'height': 0.02},
       'ice':       {'roughness': 0.0, 'slipperiness': 100.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0},
       'sand':      {'roughness': 1.0, 'slipperiness': 0.1, 'hardness': 30.0, 'elasticity': 0.0, 'height': 0.02},
       'gravel':    {'roughness': 7.0, 'slipperiness': 0.1, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.03},
       'grass':     {'roughness': 5.0, 'slipperiness': 0.0, 'hardness': 30.0, 'elasticity': 0.6, 'height': 0.05},
       'swamp':     {'roughness': 0.0, 'slipperiness': 5.0, 'hardness': 0.0, 'elasticity': 0.0, 'height': 0.1},
       'rock':      {'roughness': 10.0, 'slipperiness': 0.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.1},
       'tiles':     {'roughness': 5.0, 'slipperiness': 30.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0},
       'snow':      {'roughness': 0.0, 'slipperiness': 80.0, 'hardness': 20.0, 'elasticity': 0.0, 'height': 0.02},
       'rubber':    {'roughness': 8.0, 'slipperiness': 0.0, 'hardness': 80.0, 'elasticity': 2.0, 'height': 0.0},
       'carpet':    {'roughness': 3.0, 'slipperiness': 0.0, 'hardness': 40.0, 'elasticity': 0.3, 'height': 0.02},
       'wood':      {'roughness': 6.0, 'slipperiness': 0.0, 'hardness': 80.0, 'elasticity': 0.2, 'height': 0.02},
       'plastic':   {'roughness': 1.0, 'slipperiness': 2.0, 'hardness': 60.0, 'elasticity': 1.0, 'height': 0.0},
       'foam':      {'roughness': 5.0, 'slipperiness': 0.0, 'hardness': 0.0, 'elasticity': 2.0, 'height': 0.07}}

noise_types = {'no_noise': ('nn_', 0.0), 'noise_5p': ('n5p_', 0.05), 'noise_10p': ('n10p_', 0.1),
               'noise_20p': ('n20p_', 0.2)}

if __name__ == '__main__':
    params = args.params

    with open('../cache/params/terrain_types.json', 'w') as f:
        json.dump(terrain_types, f)

    with open('../cache/params/terrain_qualities.json', 'w') as f:
        json.dump(terrain_qualities, f)

    with open('../cache/params/qualities_ranges.json', 'w') as f:
        json.dump(qualities_ranges, f)

    with open('../cache/params/env.json', 'w') as f:
        json.dump(env, f)

    with open('../cache/params/noise_types.json', 'w') as f:
        json.dump(noise_types, f)