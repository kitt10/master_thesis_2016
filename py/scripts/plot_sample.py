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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from random import choice
from scripts.functions import load_params, read_data, f_range_gen, norm_signal, add_signal_noise


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plots sensory data.')
    parser.add_argument('-s', '--sensors', type=str, default=all_sensors, nargs='+', choices=all_sensors,
                        help='Sensors to be involved (integers)')
    parser.add_argument('-t', '--terrains', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
                        nargs='+', choices=range(1, 16), help='Terrains to be involved (integers)')
    parser.add_argument('-n', '--noises', type=str, nargs='+', default=noise_types, choices=noise_types,
                        help='Terrain noises to be involved')
    parser.add_argument('-sl', '--sample_len', type=int, default=40, choices=range(1, 85),
                        help='Number of timesteps for each sensor')
    parser.add_argument('-nr', '--norm', type=bool, default=False,
                        help='Whether to normalize the signal')
    parser.add_argument('-sn', '--signal_noise', type=float, choices=list(f_range_gen(start=0.0, stop=1.0, step=0.005)),
                        default=0.0, help='Signal noise std')
    parser.add_argument('-r', '--range', type=int, default=[0, 2400], nargs=2, choices=range(2500),
                        help='Range of the feature vector to be plotted')
    parser.add_argument('-ns', '--n_samples', type=int, default=500, choices=range(501),
                        help='Number of samples per terrain')
    parser.add_argument('-saf', '--save_fig', type=bool, default=False,
                        help='Whether to save the results')
    parser.add_argument('-shf', '--show_fig', type=bool, default=False,
                        help='Whether to show the figure')
    parser.add_argument('-na', '--noise_analysis', type=str, default=[], nargs='+', choices=['tsv', 'tcv', 'ssv', 'scv', 'ssm', 'tsm'],
                        help='tsv: terrain samples variance, tcv: terrain classes variance, ssv: signal samples var..')
    return parser.parse_args()


def prepare_signal(signal, sen):
    """
    :param signal: raw signal
    :param sen: sensor used to measure this signal
    :return: normalized signal with a signal noise
    """
    global norm, signal_noise_std

    if norm:
        ''' First, normalize the signal '''
        normed_signal = norm_signal(signal=signal, the_min=sensors_ranges[sen][0], the_max=sensors_ranges[sen][1])

        ''' Adding signal noise of defined std '''
        noised_signal = add_signal_noise(signal=normed_signal, std=signal_noise_std)
        return noised_signal
    else:
        return signal


if __name__ == '__main__':
    all_sensors, sensors_ranges, terrain_types, noise_types, noise_params, env = \
        load_params('sensors', 'sensors_ranges', 'terrain_types', 'noise_types', 'noise_params', 'env')
    args = parse_arguments()

    noises_to_use = args.noises
    terrains_to_use = [terrain_types[str(i)] for i in sorted(args.terrains)]
    sensors_to_use = args.sensors
    sample_len = args.sample_len
    norm = args.norm
    signal_noise_std = args.signal_noise
    r = args.range
    n_samples = args.n_samples
    save_figure = args.save_fig
    show_figure = args.show_fig
    noise_analysis = args.noise_analysis

    data = read_data(noises=noises_to_use, terrains=terrains_to_use, sensors=sensors_to_use, n_samples=n_samples)

    print '\n\n ## Cutting samples, normalizing and adding a signal noise...'
    samples = dict()
    for noise in noises_to_use:
        samples[noise] = dict()
        for terrain in terrains_to_use:
            samples[noise][terrain] = [[] for i in range(len(data[noise][terrain][sensors_to_use[0]]))]
            for sensor in sensors_to_use:
                for i_sample, sample_terrain in enumerate(data[noise][terrain][sensor]):
                    samples[noise][terrain][i_sample] += prepare_signal(signal=sample_terrain[10:sample_len + 10], sen=sensor)

    if not noise_analysis:
        batch_steps = np.arange(start=0, stop=len(sensors_to_use) * sample_len, step=sample_len)
        for noise in noises_to_use:
            fig = plt.figure('plot_sample_' + noise+'_'+str(signal_noise_std)+'_'+str(sample_len), figsize=(12, 7))
            plt.gcf().subplots_adjust(bottom=0.2)
            for terrain in terrains_to_use:
                for batch in [np.arange(start=step, stop=step+sample_len) for step in batch_steps]:
                    do_it = False
                    if batch[0] < r[0]:
                        if batch[1] > r[0]:
                            batch[0] = r[0]
                            do_it = True
                    if batch[1] > r[1]:
                        if batch[0] < r[1]:
                            batch[1] = r[1]
                            do_it = True
                    if batch[0] >= r[0] and batch[1] <= r[1]:
                        do_it = True

                    if do_it:
                        if batch[0] == 0:
                            #plt.plot(batch, np.mean(samples[noise][terrain], axis=0)[batch[0]:batch[-1] + 1], color=env[terrain]['color'], label=terrain)
                            plt.plot(batch, choice(samples[noise][terrain])[batch[0]:batch[-1] + 1], color=env[terrain]['color'], label=terrain)
                        else:
                            #plt.plot(batch, np.mean(samples[noise][terrain], axis=0)[batch[0]:batch[-1] + 1], color=env[terrain]['color'])
                            plt.plot(batch, choice(samples[noise][terrain])[batch[0]:batch[-1] + 1], color=env[terrain]['color'])
                            '''for sample in samples[noise][terrain]:
                                plt.plot(batch, sample[batch[0]:batch[-1] + 1], color=env[terrain]['color'])'''
            '''plt.title('Terrain Classification for AMOS II : Feature Vector : Mean of 500 samples')
            plt.suptitle('timesteps: '+str(sample_len)+', normed: '+str(norm)+', signal noise: '+str(signal_noise_std)+
                      ', terrain noise: '+str(noise)+', zoom: '+str(r))'''
            plt.xlabel('timesteps sensor by sensor')
            plt.ylabel('sensor values')
            plt.grid()
            plt.legend(loc='lower left', prop={'size': 12}, ncol=5)
            ax = fig.add_subplot(111)
            for i, sensor in enumerate(sensors_to_use):
                if r[0] <= (i+1)*sample_len <= r[1]:
                    plt.annotate(sensor, xy=(i*sample_len+0.5*sample_len, -0.1-(i%2)*0.15), color='#6C0505', horizontalalignment='center')
                    plt.plot(((i+1)*sample_len, (i+1)*sample_len), (-0.5, 1), '-..', color='#6C0505')
            if save_figure:
                plt.savefig('../../results/png/plot_sample_'+noise+'_'+str(signal_noise_std)+'_'+str(sample_len)+'_'+str(r)+'.png', bbox_inches='tight', pad_inches=0.1)
                plt.savefig('../../results/eps/plot_sample_'+noise+'_'+str(signal_noise_std)+'_'+str(sample_len)+'_'+str(r)+'.eps', bbox_inches='tight', pad_inches=0.1)
            if show_figure:
                plt.show()
            plt.close()
    else:
        if 'tsv' in noise_analysis:
            # Terrain Noise Analysis : Samples Variance
            colors = ('r', 'g', 'b', 'y', 'm', 'c', 'k')
            for terrain in terrains_to_use:
                fig = plt.figure('terrain_noise_analysis_' + terrain + '_' + str(sample_len), figsize=(12, 7))
                sigs = list()
                for noise, color in zip(noises_to_use, colors):
                    sigs.append(np.var(samples[noise][terrain], axis=0)[r[0]:r[1]])
                    plt.plot(range(r[0], r[1]), sigs[-1], label=noise, color=color)
                '''
                plt.title('Terrain Classification for AMOS II : Terrain Noise Influence : Samples Variance for terrain '+terrain)
                plt.suptitle('timesteps: ' + str(sample_len) + ', no signal noise, 500 samples, feature vector zoom: ' + str(r) + ' => foot contact sensors')
                '''
                plt.xlabel('timesteps sensor by sensor')
                plt.ylabel('samples variance')
                plt.legend(loc='best')
                plt.grid()
                ax = fig.add_subplot(111)
                m = np.max(sigs)
                for i, sensor in enumerate(sensors_to_use):
                    if r[0] <= (i + 1) * sample_len <= r[1]:
                        plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, -m/10.0 + (i % 2) * m/50.0), color='#6C0505',
                                     horizontalalignment='center')
                        plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (-m/10.0-m/50.0, m+m/50.0), '-..', color='#6C0505')
                if save_figure:
                    plt.savefig('../../results/png/tn_analysis_' + str(sample_len) + '_' + str(r) + '_'+terrain+'.png',
                                bbox_inches='tight', pad_inches=0.1)
                    plt.savefig('../../results/eps/tn_analysis_' + str(sample_len) + '_' + str(r) + '_'+terrain+'.eps',
                                bbox_inches='tight', pad_inches=0.1)
                if show_figure:
                    plt.show()

        if 'tcv' in noise_analysis:
            # Terrain Noise Analysis : Classes Variance
            means = dict()
            fig = plt.figure('terrain_noise_analysis_'+str(signal_noise_std) + '_' + str(sample_len), figsize=(12, 7))
            for noise in noises_to_use:
                means[noise] = list()
                for terrain in terrains_to_use:
                    means[noise].append(np.mean(samples[noise][terrain], axis=0))

                plt.plot(range(r[0], r[1]), np.var(means[noise], axis=0)[r[0]:r[1]], label=noise)

            plt.title('Terrain Classification for AMOS II : Terrain Noise Influence : Classes Variance')
            plt.suptitle('timesteps: ' + str(sample_len) + ', no signal noise, zoom: ' + str(r) + ' => angle sensors')
            plt.xlabel('timesteps sensor by sensor')
            plt.ylabel('var')
            plt.legend(loc='best')
            plt.grid()
            ax = fig.add_subplot(111)
            for i, sensor in enumerate(sensors_to_use):
                if r[0] <= (i + 1) * sample_len <= r[1]:
                    plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, 0.0028 - (i % 2) * 0.0002), color='#6C0505',
                                 horizontalalignment='center')
                    plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (0.0, 0.003), '-..', color='#6C0505')
                    '''
                    plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, 0.19 - (i % 2) * 0.01), color='#6C0505',
                                 horizontalalignment='center')
                    plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (0.0, 0.2), '-..', color='#6C0505')
                    '''
            if save_figure:
                plt.savefig('../../results/png/tn_analysis_' + str(sample_len) + '_' + str(r) + '.png', bbox_inches='tight',
                            pad_inches=0.1)
                plt.savefig('../../results/eps/tn_analysis_' + str(sample_len) + '_' + str(r) + '.eps', bbox_inches='tight',
                            pad_inches=0.1)
            if show_figure:
                plt.show()

        if 'ssv' in noise_analysis:
            # Signal Noise Analysis : Samples Variance

            noises_to_use = (0.0, 0.01, 0.03, 0.1)
            samples = dict()
            for s_noise in noises_to_use:
                samples[s_noise] = dict()
                signal_noise_std = s_noise
                for terrain in terrains_to_use:
                    samples[s_noise][terrain] = [[] for i in range(len(data['no_noise'][terrain][sensors_to_use[0]]))]
                    for sensor in sensors_to_use:
                        for i_sample, sample_terrain in enumerate(data['no_noise'][terrain][sensor]):
                            samples[s_noise][terrain][i_sample] += prepare_signal(signal=sample_terrain[10:sample_len + 10], sen=sensor)

            colors = ('r', 'g', 'b', 'y', 'm', 'c', 'k')
            for terrain in terrains_to_use:
                fig = plt.figure('signal_noise_analysis_' + terrain + '_' + str(sample_len), figsize=(12, 7))
                sigs = list()
                for noise, color in zip(noises_to_use, colors):
                    sigs.append(np.var(samples[noise][terrain], axis=0)[r[0]:r[1]])
                    plt.plot(range(r[0], r[1]), sigs[-1], label=noise, color=color)
                plt.title(
                    'Terrain Classification for AMOS II : Signal Noise Influence : Samples Variance for terrain ' + terrain)
                plt.suptitle(
                    'timesteps: ' + str(sample_len) + ', no terrain noise, 500 samples, feature vector zoom: ' + str(
                        r) + ' => angle sensors')
                plt.xlabel('timesteps sensor by sensor')
                plt.ylabel('variance')
                plt.legend(loc='best')
                plt.grid()
                ax = fig.add_subplot(111)
                m = np.max(sigs)
                for i, sensor in enumerate(sensors_to_use):
                    if r[0] <= (i + 1) * sample_len <= r[1]:
                        plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, -m / 10.0 + (i % 2) * m / 50.0),
                                     color='#6C0505',
                                     horizontalalignment='center')
                        plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (-m / 10.0 - m / 50.0, m + m / 50.0),
                                 '-..', color='#6C0505')
                if save_figure:
                    plt.savefig(
                        '../../results/png/sn_analysis_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.png',
                        bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(
                        '../../results/eps/sn_analysis_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.eps',
                        bbox_inches='tight', pad_inches=0.1)
                if show_figure:
                    plt.show()

        if 'scv' in noise_analysis:
            # Signal Noise Analysis : Classes Variance

            noises_to_use = (0.0, 0.01, 0.03, 0.1)
            samples = dict()
            for s_noise in noises_to_use:
                samples[s_noise] = dict()
                signal_noise_std = s_noise
                for terrain in terrains_to_use:
                    samples[s_noise][terrain] = [[] for i in range(len(data['no_noise'][terrain][sensors_to_use[0]]))]
                    for sensor in sensors_to_use:
                        for i_sample, sample_terrain in enumerate(data['no_noise'][terrain][sensor]):
                            samples[s_noise][terrain][i_sample] += prepare_signal(
                                signal=sample_terrain[10:sample_len + 10], sen=sensor)

            means = dict()
            fig = plt.figure('signal_noise_analysis_classes_variance_' + str(sample_len), figsize=(12, 7))
            for s_noise in noises_to_use:
                means[s_noise] = list()
                for terrain in terrains_to_use:
                    means[s_noise].append(np.mean(samples[s_noise][terrain], axis=0))

                plt.plot(range(r[0], r[1]), np.var(means[s_noise], axis=0)[r[0]:r[1]], label=s_noise)

            plt.title('Terrain Classification for AMOS II : Signal Noise Influence : Classes Variance')
            plt.suptitle('timesteps: ' + str(sample_len) + ', no terrain noise, feature vector zoom: ' + str(r) + ' => foot_contact sensors')
            plt.xlabel('timesteps sensor by sensor')
            plt.ylabel('variance')
            plt.legend(loc='best')
            plt.grid()
            ax = fig.add_subplot(111)
            for i, sensor in enumerate(sensors_to_use):
                if r[0] <= (i + 1) * sample_len <= r[1]:
                    '''
                    plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, 0.0043 - (i % 2) * 0.0002),
                                 color='#6C0505',
                                 horizontalalignment='center')
                    plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (0.0, 0.0045), '-..', color='#6C0505')
                    '''
                    plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, 0.19 - (i % 2) * 0.01), color='#6C0505',
                                 horizontalalignment='center')
                    plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (0.0, 0.2), '-..', color='#6C0505')

            if save_figure:
                plt.savefig('../../results/png/sn_analysis_' + str(sample_len) + '_' + str(r) + '.png',
                            bbox_inches='tight', pad_inches=0.1)
                plt.savefig('../../results/eps/sn_analysis_' + str(sample_len) + '_' + str(r) + '.eps',
                            bbox_inches='tight', pad_inches=0.1)
            if show_figure:
                plt.show()

        if 'ssm' in noise_analysis:

            # Signal Noise Analysis : Samples means for various signal noises

            noises_to_use = (0.1, 0.05, 0.03, 0.01, 0.0)
            samples = dict()
            for s_noise in noises_to_use:
                samples[s_noise] = dict()
                signal_noise_std = s_noise
                for terrain in terrains_to_use:
                    samples[s_noise][terrain] = [[] for i in range(len(data['no_noise'][terrain][sensors_to_use[0]]))]
                    for sensor in sensors_to_use:
                        for i_sample, sample_terrain in enumerate(data['no_noise'][terrain][sensor]):
                            samples[s_noise][terrain][i_sample] += prepare_signal(
                                signal=sample_terrain[10:sample_len + 10], sen=sensor)

            colors = ('#cccccc', 'y', 'g', 'b', 'r', 'm', 'c', 'k')
            for terrain in terrains_to_use:
                fig = plt.figure('signal_noise_analysis_' + terrain + '_' + str(sample_len), figsize=(12, 7))
                sigs = list()
                for noise, color in zip(noises_to_use, colors):
                    #sigs.append(np.mean(samples[noise][terrain], axis=0)[r[0]:r[1]])
                    sigs.append(choice(samples[noise][terrain])[r[0]:r[1]])
                    plt.plot(range(r[0], r[1]), sigs[-1], label='std: '+str(noise), color=color)
                '''
                plt.title('Terrain Classification for AMOS II : Signal Noise Influence on One Sample for Terrain ' + terrain)
                plt.suptitle('timesteps: ' + str(sample_len) + ', no terrain noise, 500 samples, feature vector zoom: ' + str(
                        r) + ' => angle sensors')'''
                plt.xlabel('timesteps sensor by sensor')
                plt.ylabel('sensor value')
                plt.legend(loc='best')
                plt.grid()
                ax = fig.add_subplot(111)
                m = np.max(sigs)
                for i, sensor in enumerate(sensors_to_use):
                    if r[0] <= (i + 1) * sample_len <= r[1]:
                        plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, -m / 10.0 + (i % 2) * m / 50.0),
                                     color='#6C0505',
                                     horizontalalignment='center')
                        plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (-m / 10.0 - m / 50.0, m + m / 50.0),
                                 '-..', color='#6C0505')
                if save_figure:
                    plt.savefig(
                        '../../results/png/sn_analysis_mean_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.png',
                        bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(
                        '../../results/eps/sn_analysis_mean_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.eps',
                        bbox_inches='tight', pad_inches=0.1)
                if show_figure:
                    plt.show()

        if 'tsm' in noise_analysis:

            # Terrain Noise Analysis : Samples for various terrain noises

            mpl.rcParams['axes.labelsize'] = 18
            mpl.rcParams['xtick.labelsize'] = 15
            mpl.rcParams['ytick.labelsize'] = 15
            mpl.rcParams['legend.fontsize'] = 18

            #noises_to_use = noise_types
            samples = dict()
            for noise in noises_to_use:
                samples[noise] = dict()
                terrain_noise_std = noise_params[noise][1]
                for terrain in terrains_to_use:
                    samples[noise][terrain] = [[] for i in range(len(data[noises_to_use[0]][terrain][sensors_to_use[0]]))]
                    for sensor in sensors_to_use:
                        for i_sample, sample_terrain in enumerate(data[noises_to_use[0]][terrain][sensor]):
                            samples[noise][terrain][i_sample] += prepare_signal(
                                signal=sample_terrain[10:sample_len + 10], sen=sensor)

            colors = ('r', 'b', 'k', '#cccccc', 'y', 'g', 'm', 'c')
            for terrain in terrains_to_use:
                fig = plt.figure('signal_noise_analysis_' + terrain + '_' + str(sample_len), figsize=(12, 7))
                sigs = list()
                for noise, color in zip(noises_to_use, colors):
                    #sigs.append(np.mean(samples[noise][terrain], axis=0)[r[0]:r[1]])
                    sigs.append(choice(samples[noise][terrain])[r[0]:r[1]])
                    plt.plot(range(r[0], r[1]), sigs[-1], label=noise, color=color)
                '''plt.title(
                    'Terrain Classification for AMOS II : Signal Noise Influence on One Sample for Terrain ' + terrain)
                plt.suptitle(
                    'timesteps: ' + str(sample_len) + ', no terrain noise, 500 samples, feature vector zoom: ' + str(
                        r) + ' => angle sensors')'''
                plt.xlabel('timesteps sensor by sensor')
                plt.ylabel('mean sensor value')
                plt.legend(loc='best')
                plt.grid()
                ax = fig.add_subplot(111)
                m = np.max(sigs)
                for i, sensor in enumerate(sensors_to_use):
                    if r[0] <= (i + 1) * sample_len <= r[1]:
                        plt.annotate(sensor, xy=(i * sample_len + 0.5 * sample_len, -m / 15.0 + (i % 2) * m / 40.0),
                                     color='#6C0505',
                                     horizontalalignment='center')
                        plt.plot(((i + 1) * sample_len, (i + 1) * sample_len), (-m / 15.0 - m / 40.0, m + m / 40.0),
                                 '-..', color='#6C0505')
                if save_figure:
                    plt.savefig(
                        '../../results/png/sn_analysis_mean_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.png',
                        bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(
                        '../../results/eps/sn_analysis_mean_' + str(sample_len) + '_' + str(r) + '_' + terrain + '.eps',
                        bbox_inches='tight', pad_inches=0.1)
                if show_figure:
                    plt.show()