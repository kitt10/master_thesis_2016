# -*- coding: utf-8 -*-
"""
    scripts.functions
    ~~~~~~~~~~~~~~~~~

    Not a script. Just 'static' common functions used in other scripts.
"""

import json
from glob import glob
from os import path
from numpy import random
from shelve import open as open_shelve


def load_params(*params_names):
    params = list()
    for param_name in params_names:
        with open('../cache/params/'+param_name+'.json') as f:
            params.append(json.load(f))
    return params


def f_range_gen(start, stop, step):
    while start <= stop:
        yield round(start, 2)
        start += step


def norm(value, the_max):
    return value/the_max


def norm_signal(signal, the_min, the_max):
    return [min(max(float((x-the_min))/(the_max-the_min), 0), 1) for x in signal]


def read_data(noises, terrains, sensors, n_samples=10):
    noise_params = load_params('noise_params')[0]
    data = dict()

    for noise in noises:
        data[noise] = dict()
        for terrain in terrains:
            data[noise][terrain] = dict()
            data[noise][terrain]['data_str'] = list()
            for sensor in sensors:
                data[noise][terrain][sensor] = list()
            txt_samples = glob(path.join('../../data/'+noise+'/'+noise_params[noise][0]+terrain, '*.txt'))
            for i_sample, path_and_filename in enumerate(sorted(txt_samples)):
                if i_sample >= n_samples:
                    break
                with open(path_and_filename, 'r') as data_file:
                    data[noise][terrain]['data_str'].append(data_file.read())
                for i_sensor, sensor in enumerate(sensors):
                    data[noise][terrain][sensor].append([0.0])
                    for line in data[noise][terrain]['data_str'][-1].split('\n')[:95]:
                        values = line.split(';')
                        data[noise][terrain][sensor][i_sample].append(float(values[i_sensor + 1]))
            print 'Data for', noise, terrain, 'found (' + str(len(data[noise][terrain][sensors[0]])) + ' samples).'

    return data


def load_net(destination):
    net_file = open_shelve(destination, 'r')
    net = dict(net_file)
    net_file.close()
    return net


def add_signal_noise(signal, std):
    """
    :param signal: normed clean signal
    :return: noised signal (Gaussian noise of zero mean and defined std)
    """
    signal_noise_std = 1e-10 if std <= 0 else std
    noise = random.normal(loc=0, scale=signal_noise_std, size=len(signal))
    return [min(max(x+n, 0.0), 1.0) for x, n in zip(signal, noise)]
