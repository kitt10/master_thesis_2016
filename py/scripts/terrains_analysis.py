#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.terrains_analysis
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This script shows estimated parameters of all virtual terrains.
    Additionally it computes similarities among these terrains.
    Outputs are presented as "matshow" plots.

    @arg terrains       : Terrains to be plotted.
"""

import argparse
import json
import matplotlib.pyplot as plt

__author__ = 'Martin Bulin'
__copyright__ = 'Copyright 2016, Master Thesis'
__credits__ = ['Martin Bulin', 'Tomas Kulvicius', 'Poramate Manoonpong']
__license__ = 'GPL'
__version__ = '1.0'
__maintainer__ = 'Martin Bulin'
__email__ = 'bulinmartin@gmail.com'
__status__ = 'Development'


parser = argparse.ArgumentParser(description='Plots chosen terrains parameters and makes a simple analysis.')
parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                    help='Terrains to be plotted (integers)')
args = parser.parse_args()


def norm(value, q):
    return value/ranges[q][1]


if __name__ == '__main__':

    with open('../cache/params/terrain_types.json') as f:
        terrains = json.load(f)

    with open('../cache/params/terrain_qualities.json') as f:
        qualities = json.load(f)

    with open('../cache/params/qualities_ranges.json') as f:
        ranges = json.load(f)

    with open('../cache/params/env.json') as f:
        env = json.load(f)

    terrains_to_use = [terrains[str(i)] for i in sorted(args.terrains)]

    ''' Terrains Parameters '''
    plt.matshow([[norm(env[terrain][quality], quality) for quality in qualities] for terrain in terrains_to_use],
                vmin=0.0, vmax=1.0)
    plt.xticks(range(len(qualities)), qualities, rotation=45)
    plt.yticks(range(len(terrains_to_use)), terrains_to_use)
    plt.colorbar()
    plt.suptitle('Chosen Terrains Parameters')
    for q_i, quality in enumerate(qualities):
        for t_i, terrain in enumerate(terrains_to_use):
            plt.text(q_i, t_i, norm(env[terrain][quality], quality), va='center', ha='center')
    plt.show()

    ''' Terrains Variability '''
    distances = dict()
    for terrain1 in terrains_to_use:
        distances[terrain1] = dict()
        for terrain2 in terrains_to_use:
            distances[terrain1][terrain2] = 0.0
            for quality in qualities:
                distances[terrain1][terrain2] += abs(norm(env[terrain1][quality], quality)-norm(env[terrain2][quality], quality))
            #print terrain1, terrain2, distances[terrain1][terrain2]

    plt.matshow([[distances[terrain1][terrain2] for terrain2 in terrains_to_use] for terrain1 in terrains_to_use],
                vmin=0.0, vmax=5.0)
    plt.xticks(range(len(terrains_to_use)), terrains_to_use, rotation=45)
    plt.yticks(range(len(terrains_to_use)), terrains_to_use)
    plt.colorbar()
    plt.suptitle('Terrains Variability')
    for t1_i, terrain1 in enumerate(terrains_to_use):
        for t2_i, terrain2 in enumerate(terrains_to_use):
            plt.text(t1_i, t2_i, round(distances[terrain1][terrain2], 2), va='center', ha='center')
    plt.show()