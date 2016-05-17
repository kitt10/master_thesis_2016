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
import matplotlib.pyplot as plt
from functions import load_params, norm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots chosen terrains parameters and makes a simple analysis.')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be plotted (integers)')
    return parser.parse_args()


if __name__ == '__main__':
    terrains, qualities, ranges, env = load_params('terrain_types', 'terrain_qualities', 'qualities_ranges', 'env')
    args = parse_arguments()

    terrains_to_use = [terrains[str(i)] for i in sorted(args.terrains)]

    ''' Terrains Parameters '''
    plt.matshow([[norm(env[terrain][quality], ranges[quality][1]) for terrain in terrains_to_use] for quality in qualities],
                vmin=0.0, vmax=1.0)
    plt.xticks(range(len(terrains_to_use)), terrains_to_use, rotation=45)
    plt.yticks(range(len(qualities)), qualities)
    plt.colorbar()
    plt.suptitle('Chosen Terrains Parameters')
    for t_i, terrain in enumerate(terrains_to_use):
        for q_i, quality in enumerate(qualities):
            plt.text(t_i, q_i, norm(env[terrain][quality], ranges[quality][1]), va='center', ha='center')
    plt.savefig('../../results/png/terrains_parameters.png', bbox_inches='tight')
    plt.savefig('../../results/eps/terrains_parameters.eps', bbox_inches='tight')
    plt.show()

    ''' Terrains Variability '''
    distances = dict()
    for terrain1 in terrains_to_use:
        distances[terrain1] = dict()
        for terrain2 in terrains_to_use:
            distances[terrain1][terrain2] = 0.0
            for quality in qualities:
                distances[terrain1][terrain2] += \
                    abs(norm(env[terrain1][quality], ranges[quality][1])-norm(env[terrain2][quality], ranges[quality][1]))
            #print terrain1, terrain2, distances[terrain1][terrain2]

    plt.matshow([[distances[terrain1][terrain2]/5.0 for terrain2 in terrains_to_use] for terrain1 in terrains_to_use],
                vmin=0.0, vmax=1.0)
    plt.xticks(range(len(terrains_to_use)), terrains_to_use, rotation=45)
    plt.yticks(range(len(terrains_to_use)), terrains_to_use)
    plt.colorbar()
    #plt.suptitle('Terrains Mutual Similarity Factors', fontsize=15)
    for t1_i, terrain1 in enumerate(terrains_to_use):
        for t2_i, terrain2 in enumerate(terrains_to_use):
            plt.text(t1_i, t2_i, round(distances[terrain1][terrain2]/5.0, 1), va='center', ha='center', fontsize=10)
    plt.savefig('../../results/png/terrains_variability.png', bbox_inches='tight')
    plt.savefig('../../results/eps/terrains_variability.eps', bbox_inches='tight')
    plt.savefig('../../thesis/img/terrains_variability.eps', bbox_inches='tight')
    plt.show()
