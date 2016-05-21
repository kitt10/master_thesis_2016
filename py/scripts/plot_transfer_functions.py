#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from kitt_nn.nn_tool.nn_function import sigmoid, tanh

''' customizing mpl '''
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 25

x = np.arange(-100, 100, 0.01)
plt.plot(x, sigmoid(x), '2-', label='f(z): sigmoid')
plt.plot(x, tanh(x), '2-', label='tanh(z)')
plt.plot((-15, 15), (1, 1), '--', color='maroon')
plt.plot((-15, 15), (-1, -1), '--', color='maroon')
plt.plot((-15, 15), (0, 0), '--', color='maroon')
plt.xlabel('neuron activation z')
plt.ylabel('neuron activity a')
plt.xlim([-15, 15])
plt.ylim([-1.5, 1.5])
plt.grid()
plt.legend(loc='best')
plt.savefig('../../thesis/img/transfer_functions.eps', bbox_inches='tight', pad_inches=0.1)
plt.show()