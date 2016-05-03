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
from scripts.functions import load_params, read_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots sensory data.')
    parser.add_argument('-s', '--sensors', type=str, default=all_sensors, nargs='+', choices=all_sensors,
                        help='Sensors to be involved (integers)')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be involved (integers)')
    parser.add_argument('-n', '--noises', type=str, nargs='+', default=noise_types, choices=noise_types,
                        help='Terrain noises to be involved')
    return parser.parse_args()

if __name__ == '__main__':
    all_sensors, sensors_ranges, terrain_types, noise_types, noise_params, env = \
        load_params('sensors', 'sensors_ranges', 'terrain_types', 'noise_types', 'noise_params', 'env')
    args = parse_arguments()

    noises_to_use = args.noises
    terrains_to_use = [terrain_types[str(i)] for i in sorted(args.terrains)]
    sensors_to_use = args.sensors

    data = read_data(noises=noises_to_use, terrains=terrains_to_use, sensors=sensors_to_use)

    for sensor in sensors_to_use:
        plt.figure('plot_sensor.py : '+sensor, figsize=(7, 3))
        plt.gcf().subplots_adjust(bottom=0.2)
        for noise in noises_to_use:
            for terrain in terrains_to_use:
                plt.plot(np.mean(data[noise][terrain][sensor], axis=0), color=env[terrain]['color'], label=noise+'/'+terrain)

        plt.title('AMOS II sensor: '+sensor)
        plt.xlabel('timesteps')
        plt.ylabel('sensor values')
        plt.grid()
        plt.ylim(sensors_ranges[sensor])
        plt.legend()
        plt.show()
