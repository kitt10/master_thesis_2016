# -*- coding: utf-8 -*-
"""
    scripts.functions
    ~~~~~~~~~~~~~~~~~

    Not a script. Just 'static' common functions used in other scripts.
"""

import json


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
