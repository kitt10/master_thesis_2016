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
from glob import glob
from os import path
from numpy.random import normal
from shelve import open as open_shelve
from time import gmtime, strftime
from functions import load_params, f_range_gen, norm_signal


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates a .pkl dataset from cleaned .txt data.')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be involved (integers)')
    parser.add_argument('-tn', '--terrain_noise', type=str, default='no_noise', choices=noise_types,
                        help='Terrain noise to be used')
    parser.add_argument('-sn', '--signal_noise', type=float,
                        default=0.0, choices=list(f_range_gen(start=0.0, stop=0.1, step=0.005)),
                        help='Signal noise standard deviation (percentage)')
    parser.add_argument('-s', '--sensors', type=str, nargs='+', default=all_sensors, choices=all_sensors,
                        help='Sensors to be used for classification')
    parser.add_argument('-sl', '--sample_len', type=int, default=80,
                        help='Length of one sample (simulation timesteps)')
    parser.add_argument('-ds', '--data_split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        choices=list(f_range_gen(start=0.0, stop=1.0, step=0.01)),
                        help='Training : Validation : Testing data split')
    parser.add_argument('-dn', '--destination_name', type=str, default=strftime('dataset_%Y_%m_%d_%H_%M_%S', gmtime()),
                        help='Length of one sample (simulation timesteps)')
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

    ''' First, normalize the signal '''
    normed_signal = norm_signal(signal=signal, the_min=sensors_ranges[sen][0], the_max=sensors_ranges[sen][1])

    ''' Adding signal noise of defined std '''
    noised_signal = add_signal_noise(signal=normed_signal)
    return noised_signal


def add_signal_noise(signal):
    """
    :param signal: normed clean signal
    :return: noised signal (Gaussian noise of zero mean and defined std)
    """
    global signal_noise_std
    signal_noise_std = 1e-10 if signal_noise_std <= 0 else signal_noise_std
    noise = normal(loc=0, scale=signal_noise_std, size=len(signal))
    return [x+n for x, n in zip(signal, noise)]


if __name__ == '__main__':
    terrain_types, all_sensors, sensors_ranges, noise_types, noise_params = \
        load_params('terrain_types', 'sensors', 'sensors_ranges', 'noise_types', 'noise_params')
    args = parse_arguments()

    terrains_to_use = [terrain_types[str(i)] for i in sorted(args.terrains)]
    sensors_to_use = args.sensors
    terrain_noise = args.terrain_noise
    signal_noise_std = args.signal_noise
    sample_len = args.sample_len
    data_split = args.data_split
    destination_name = '../cache/datasets/amos_terrains_sim/'+args.destination_name+'.ds'

    ''' Reading .txt files to a dict called data '''
    print '\n\n ## Reading .txt data files...'
    data = dict()
    for terrain in terrains_to_use:
        data[terrain] = dict()
        data[terrain]['data_str'] = list()
        for sensor in sensors_to_use:
            data[terrain][sensor] = list()
        txt_samples = glob(path.join('../../data/'+terrain_noise+'/'+noise_params[terrain_noise][0]+terrain, '*.txt'))
        for i_sample, path_and_filename in enumerate(sorted(txt_samples)):
            with open(path_and_filename, 'r') as data_file:
                data[terrain]['data_str'].append(data_file.read())
            for i_sensor, sensor in enumerate(sensors_to_use):
                data[terrain][sensor].append([0.0])
                for line in data[terrain]['data_str'][-1].split('\n')[:95]:
                    values = line.split(';')
                    data[terrain][sensor][i_sample].append(float(values[i_sensor+1]))
        print 'Data for', terrain, 'added ('+str(len(data[terrain][sensors_to_use[0]]))+' samples).'

    ''' Cutting samples, normalizing and adding a signal noise '''
    print '\n\n ## Cutting samples, normalizing and adding a signal noise...'
    samples = dict()
    for terrain in terrains_to_use:
        samples[terrain] = [[] for i in range(len(data[terrain][sensors_to_use[0]]))]
        for sensor in sensors_to_use:
            for i_sample, sample_terrain in enumerate(data[terrain][sensor]):
                samples[terrain][i_sample] += prepare_signal(signal=sample_terrain[10:sample_len+10], sen=sensor)
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

    dataset = open_shelve(destination_name, 'c')
    dataset['x'] = x
    dataset['y'] = y
    dataset['terrains'] = terrains_to_use
    dataset['terrain_noise'] = terrain_noise
    dataset['signal_noise_std'] = signal_noise_std
    dataset['sensors'] = sensors_to_use
    dataset['sample_len'] = sample_len
    dataset['data_split'] = data_split
    dataset['size'] = (len(x['training']), len(x['validation']), len(x['testing']))
    dataset.close()

    print 'Dataset dumped.'
