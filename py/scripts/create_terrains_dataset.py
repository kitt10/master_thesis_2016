#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.create_terrains_dataset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a .pkl dataset out of cleaned .txt files. It can also add a signal noise.

    @arg terrains           : Terrains to be added to the dataset.
    @arg terrain_noise      : Terrain noise of data to be used.
    @arg signal_noise       : Standard deviation of a signal noise to be added to the data (percentage).
    @arg sensors            : Sensors to be used.
    @sample_len             : Length of one sample
    @arg data_split         : Training : Validation : Testing ratio (two floats)
    @arg destination_name   : Name of the dataset filename
"""

import argparse
from sys import stderr
from shelve import open as open_shelve
from time import gmtime, strftime
from functions import load_params, f_range_gen, read_data, norm_signal, add_signal_noise


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates a .pkl dataset from cleaned .txt data.')
    parser.add_argument('-rt', '--rem_terrains', type=int, default=[],
                        nargs='+', choices=range(1, 16), help='Terrains to be removed (integers)')
    parser.add_argument('-tn', '--terrain_noise', type=str, default='no_noise', choices=noise_types,
                        help='Terrain noise to be used')
    parser.add_argument('-sn', '--signal_noise', type=float,
                        default=0.0, choices=list(f_range_gen(start=0.0, stop=0.1, step=0.005)),
                        help='Signal noise standard deviation (percentage)')
    parser.add_argument('-s', '--sensors', type=str, default='alls', choices=['alls', 'angle', 'foot'],
                        help='Sensors to be used for classification')
    parser.add_argument('-ts', '--timesteps', type=int, default=40,
                        help='Length of one sample (simulation timesteps)')
    parser.add_argument('-ds', '--data_split', type=float, nargs=3, default=[0.7, 0.1, 0.2  ],
                        choices=list(f_range_gen(start=0.0, stop=1.0, step=0.01)),
                        help='Training : Validation : Testing data split')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('terrains_%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='File name of the dataset')
    parser.add_argument('-ns', '--n_samples', type=int, default=500, choices=range(501),
                        help='Number of samples per terrain')
    args_tmp = parser.parse_args()

    ''' Check args '''
    if abs(sum(args_tmp.data_split)-1) > 1e-5:
        stderr.write('Error: data_split args must give 1.0 together.\n')
        exit()
    else:
        return args_tmp


def prepare_signal(signal, sen):
    """
    :param signal: raw signal
    :param sen: sensor used to measure this signal
    :return: normalized signal with a signal noise
    """

    global signal_noise_std

    ''' First, normalize the signal '''
    normed_signal = norm_signal(signal=signal, the_min=sensors_ranges[sen][0], the_max=sensors_ranges[sen][1])

    ''' Adding signal noise of defined std '''
    noised_signal = add_signal_noise(signal=normed_signal, std=signal_noise_std)
    return noised_signal


if __name__ == '__main__':
    terrain_types, all_sensors, sensors_ranges, noise_types, noise_params = \
        load_params('terrain_types', 'sensors', 'sensors_ranges', 'noise_types', 'noise_params')
    args = parse_arguments()

    terrains_to_use = [terrain_types[str(i)] for i in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15] if i not in args.rem_terrains]
    if not args.rem_terrains:
        terrains_flag = 'allt'
    else:
        terrains_flag = 'rt'
        for t_i in sorted(args.rem_terrains):
            terrains_flag += str(t_i)

    sensors_to_use = all_sensors
    if args.sensors == 'angle':
        sensors_to_use = all_sensors[:18]
    elif args.sensors == 'foot':
        sensors_to_use = all_sensors[18:]
    terrain_noise = args.terrain_noise
    signal_noise_std = args.signal_noise
    timesteps = args.timesteps
    data_split = args.data_split
    n_samples = args.n_samples

    destination_name = 'amter_'+noise_params[terrain_noise][0]+'0'+str(signal_noise_std)[2:]+'_'+str(timesteps)+\
                       '_'+args.sensors+'_'+terrains_flag+'_'+str(n_samples)
    destination = '../cache/datasets/amter/'+destination_name+'.ds'

    ''' Reading .txt files to a dict called data '''
    print '\n\n ## Reading .txt data files...'
    data = read_data(noises=[terrain_noise], terrains=terrains_to_use, sensors=sensors_to_use, n_samples=n_samples)

    ''' Cutting samples, normalizing and adding a signal noise '''
    print '\n\n ## Cutting samples, normalizing and adding a signal noise...'
    samples = dict()
    for terrain in terrains_to_use:
        samples[terrain] = [[] for i in range(len(data[terrain_noise][terrain][sensors_to_use[0]]))]
        for sensor in sensors_to_use:
            for i_sample, sample_terrain in enumerate(data[terrain_noise][terrain][sensor]):
                samples[terrain][i_sample] += prepare_signal(signal=sample_terrain[10:timesteps+10], sen=sensor)
        print 'Samples of', terrain, 'cut, normalized and noised.'

    ''' Splitting data '''
    print '\n\n ## Splitting data...'
    x = {'training': list(), 'validation': list(), 'testing': list()}
    y = {'training': list(), 'validation': list(), 'testing': list()}
    for terrain in terrains_to_use:
        bounds = (len(samples[terrain])*data_split[0], len(samples[terrain])*(data_split[0]+data_split[1]))
        for i_sample, sample in enumerate(samples[terrain]):
            if i_sample < bounds[0]:
                x['training'].append(sample)
                y['training'].append(terrain)
            elif bounds[0] <= i_sample < bounds[1]:
                x['validation'].append(sample)
                y['validation'].append(terrain)
            else:
                x['testing'].append(sample)
                y['testing'].append(terrain)

    print 'Got dataset:', len(x['training']), ':', len(x['validation']), ':', len(x['testing'])

    ''' Saving dataset '''
    print '\n\n ## Saving dataset as', destination_name

    dataset = open_shelve(destination, 'c')
    dataset['x'] = x
    dataset['y'] = y
    dataset['terrains'] = terrains_to_use
    dataset['terrain_noise'] = terrain_noise
    dataset['signal_noise_std'] = signal_noise_std
    dataset['sensors'] = sensors_to_use
    dataset['timesteps'] = timesteps
    dataset['n_samples'] = n_samples
    dataset['data_split'] = data_split
    dataset['support'] = (len(x['training']), len(x['validation']), len(x['testing']))
    dataset.close()

    print '\n## Dataset dumped. ----------------------------------- '
    print ' $ terrain noise: \t', terrain_noise
    print ' $ signal noise std: \t', signal_noise_std
    print ' $ timesteps: \t\t', timesteps
    print ' $ sensors: \t\t', args.sensors
    print ' $ feature vector len: \t', len(samples[terrains_to_use[0]][0])
    print ' $ removed terrains: \t', [terrain_types[str(i)] for i in sorted(args.rem_terrains)]
    print ' $ n samples: \t\t', n_samples
    print ' $ data splitting: \t', data_split
    print ' $ support: \t\t', (len(x['training']), len(x['validation']), len(x['testing']))
    print ' $ saved as: \t\t', destination_name
    print '## --------------------------------------------------- \n\n'

