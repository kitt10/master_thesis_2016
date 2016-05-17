#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.plotting.plot_sensor
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script plots a sensory data.

    @arg sensors        : Sensors to be involved.
    @arg terrains       : Terrains to be involved.
    @arg noises         : Terrain noises to be involved.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scripts.functions import load_params, read_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots sensory data.')
    parser.add_argument('-s', '--sensors', type=str, default=all_sensors, nargs='+', choices=all_sensors,
                        help='Sensors to be involved (integers)')
    parser.add_argument('-t', '--terrains', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
                        nargs='+', choices=range(1, 16), help='Terrains to be involved (integers)')
    parser.add_argument('-n', '--noises', type=str, nargs='+', default=noise_types, choices=noise_types,
                        help='Terrain noises to be involved')
    parser.add_argument('-ns', '--n_samples', type=int, default=500, choices=range(501),
                        help='Number of samples per terrain')
    parser.add_argument('-saf', '--save_fig', type=bool, default=False,
                        help='Whether to save the results')
    parser.add_argument('-shf', '--show_fig', type=bool, default=False,
                        help='Whether to show the figure')
    return parser.parse_args()

if __name__ == '__main__':
    all_sensors, sensors_ranges, terrain_types, noise_types, noise_params, env = \
        load_params('sensors', 'sensors_ranges', 'terrain_types', 'noise_types', 'noise_params', 'env')
    args = parse_arguments()

    noises_to_use = args.noises
    terrains_to_use = [terrain_types[str(i)] for i in sorted(args.terrains)]
    sensors_to_use = args.sensors
    n_samples = args.n_samples
    save_figure = args.save_fig
    show_figure = args.show_fig

    data = read_data(noises=noises_to_use, terrains=terrains_to_use, sensors=sensors_to_use, n_samples=n_samples)

    for sensor in sensors_to_use:
        for noise in noises_to_use:
            plt.figure('plot_sensor_'+sensor+'_'+noise, figsize=(10, 5))
            plt.gcf().subplots_adjust(bottom=0.2)

            for terrain in terrains_to_use:
                #plt.plot(np.mean(data[noise][terrain][sensor], axis=0), color=env[terrain]['color'], label=terrain)
                plt.plot(choice(data[noise][terrain][sensor]), color=env[terrain]['color'], label=terrain)
                #for sample in data[noise][terrain][sensor]:
                    #plt.plot(sample)


            #plt.suptitle('sensor: '+sensor+', terrain noise: '+noise+', no signal noise, mean of 500 samples')
            #plt.title('AMOS II Terrain Classification : Simulated Sensory Output')
            plt.xlabel('timesteps')
            plt.ylabel('sensor values')
            plt.grid()
            plt.ylim(sensors_ranges[sensor])
            plt.legend(loc='best', prop={'size': 12}, ncol=3)
            if save_figure:
                plt.savefig('../../results/png/plot_sensor_' + sensor + '_' + noise + '.png', bbox_inches='tight',
                            pad_inches=0.1)
                plt.savefig('../../results/eps/plot_sensor_' + sensor + '_' + noise + '.eps', bbox_inches='tight',
                            pad_inches=0.1)
            if show_figure:
                plt.show()
            plt.close()
