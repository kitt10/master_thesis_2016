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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Sets global project parameters and stores them as .json files.')
    parser.add_argument('-p', '--params', type=str, nargs='+',
                        default=['terrain_types', 'terrain_qualities', 'qualities_ranges', 'env', 'noise_types', 'sensors'],
                        choices=['terrain_types', 'terrain_qualities', 'qualities_ranges', 'env', 'noise_types', 'sensors'],
                        help='Params to be reset and restored')
    return parser.parse_args()


if __name__ == '__main__':
    params = parse_arguments().params

    terrain_types = {1: 'concrete', 2: 'mud', 3: 'ice', 4: 'sand', 5: 'gravel', 6: 'grass', 7: 'swamp', 8: 'rock',
                     9: 'tiles', 10: 'snow', 11: 'rubber', 12: 'carpet', 13: 'wood', 14: 'plastic', 15: 'foam'}
    with open('../cache/params/terrain_types.json', 'w') as f:
        json.dump(terrain_types, f)

    terrain_qualities = ('roughness', 'slipperiness', 'hardness', 'elasticity', 'height')
    with open('../cache/params/terrain_qualities.json', 'w') as f:
        json.dump(terrain_qualities, f)

    qualities_ranges = {'roughness': (0.0, 10.0), 'slipperiness': (0.0, 100.0), 'hardness': (0.0, 100.0),
                        'elasticity': (0.0, 2.0), 'height': (0.0, 0.1)}
    with open('../cache/params/qualities_ranges.json', 'w') as f:
        json.dump(qualities_ranges, f)

    env = {
        'concrete':  {'roughness': 10.0, 'slipperiness': 0.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0, 'color': '#9C9FA6'},
        'mud':       {'roughness': 0.5, 'slipperiness': 5.0, 'hardness': 0.5, 'elasticity': 0.5, 'height': 0.02, 'color': '#646464'},
        'ice':       {'roughness': 0.0, 'slipperiness': 100.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0, 'color': '#D7E3FF'},
        'sand':      {'roughness': 1.0, 'slipperiness': 0.1, 'hardness': 30.0, 'elasticity': 0.0, 'height': 0.02, 'color': '#F2EE7C'},
        'gravel':    {'roughness': 7.0, 'slipperiness': 0.1, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.03, 'color': '#737F9C'},
        'grass':     {'roughness': 5.0, 'slipperiness': 0.0, 'hardness': 30.0, 'elasticity': 0.6, 'height': 0.05, 'color': '#239614'},
        'swamp':     {'roughness': 0.0, 'slipperiness': 5.0, 'hardness': 0.0, 'elasticity': 0.0, 'height': 0.1, 'color': '#324B32'},
        'rock':      {'roughness': 10.0, 'slipperiness': 0.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.1, 'color': '#6E5A3C'},
        'tiles':     {'roughness': 5.0, 'slipperiness': 30.0, 'hardness': 100.0, 'elasticity': 0.0, 'height': 0.0, 'color': '#FAC896'},
        'snow':      {'roughness': 0.0, 'slipperiness': 80.0, 'hardness': 20.0, 'elasticity': 0.0, 'height': 0.02, 'color': '#FDB0FB'},
        'rubber':    {'roughness': 8.0, 'slipperiness': 0.0, 'hardness': 80.0, 'elasticity': 2.0, 'height': 0.0, 'color': '#000000'},
        'carpet':    {'roughness': 3.0, 'slipperiness': 0.0, 'hardness': 40.0, 'elasticity': 0.3, 'height': 0.02, 'color': '#876496'},
        'wood':      {'roughness': 6.0, 'slipperiness': 0.0, 'hardness': 80.0, 'elasticity': 0.2, 'height': 0.02, 'color': '#5A4100'},
        'plastic':   {'roughness': 1.0, 'slipperiness': 2.0, 'hardness': 60.0, 'elasticity': 1.0, 'height': 0.0, 'color': '#96FABE'},
        'foam':      {'roughness': 5.0, 'slipperiness': 0.0, 'hardness': 0.0, 'elasticity': 2.0, 'height': 0.07, 'color': '#DCE696'}
    }
    with open('../cache/params/env.json', 'w') as f:
        json.dump(env, f)

    noise_types = {'no_noise': ('nn_', 0.0), 'noise_5p': ('n5p_', 0.05), 'noise_10p': ('n10p_', 0.1),
                   'noise_20p': ('n20p_', 0.2)}
    with open('../cache/params/noise_types.json', 'w') as f:
        json.dump(noise_types, f)

    sensors = ('atr_f', 'atr_m', 'atr_h', 'atl_f', 'atl_m', 'atl_h', 'acr_f', 'acr_m', 'acr_h', 'acl_f', 'acl_m', 'acl_h',
               'afr_f', 'afr_m', 'afr_h', 'afl_f', 'afl_m', 'afl_h', 'fr_f', 'fr_m', 'fr_h', 'fl_f', 'fl_m', 'fl_h')
    with open('../cache/params/sensors.json', 'w') as f:
        json.dump(sensors, f)
