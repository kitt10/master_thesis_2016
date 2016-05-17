#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from functions import load_net


if __name__ == '__main__':
    kitt_net = load_net('../cache/trained/kitt_mnist&0.03_10_[15].net')

    ''' Accuracy through epochs '''
    plt.figure()
    epochs = range(1, len(kitt_net['training_eval'][0]['t'])+1)
    plt.plot(epochs, kitt_net['training_eval'][0]['t'], '--o', label='kitt: training data')
    plt.plot(epochs, kitt_net['training_eval'][0]['v'], '--o', label='kitt: validation data')
    plt.xlabel('training epoch')
    plt.xlim([-1, epochs[-1] + 1])
    plt.ylabel('classification accuracy')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(loc='best', ncol=2)
    plt.show()

    ''' Error through epochs '''
    plt.figure()
    epochs = range(1, len(kitt_net['training_eval'][1]['t']) + 1)
    plt.plot(epochs, kitt_net['training_eval'][1]['t'], 'r--o', label='kitt: training data')
    plt.plot(epochs, kitt_net['training_eval'][1]['v'], 'm--o', label='kitt: validation data')
    plt.xlabel('training epoch')
    plt.xlim([-1, epochs[-1] + 1])
    plt.ylabel('classification error')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(loc='best', ncol=2)
    plt.show()

    ''' Average epoch time '''
    plt.figure()
    plt.bar(1, np.mean(kitt_net['training_eval'][2]), width=1)
    plt.xlim([0, 4])
    plt.xticks([1.5], ['kitt_nn'], ha='center')
    plt.ylabel('average epoch time [s]')
    plt.grid()
    plt.show()