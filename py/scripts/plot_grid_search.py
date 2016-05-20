#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from functions import load_net


def load():
    nets = dict()
    for lr in (0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9):
        nets[lr] = dict()
        for st in (5, 7, 10, 15, 20, 30, 50, 100):
            nets[lr][st] = dict()
            try:
                nets[lr][st]['net'] = load_net('../cache/trained/kitt_amter_nn_00_40_alls_allt_500&'+str(lr)+'_500_['+str(st)+']_1.net')

                nets[lr][st]['acc_list'] = nets[lr][st]['net']['training_eval'][0]['v']
                nets[lr][st]['err_list'] = nets[lr][st]['net']['training_eval'][1]['v']
                nets[lr][st]['time_list'] = nets[lr][st]['net']['training_eval'][2]
                nets[lr][st]['acc'] = nets[lr][st]['net']['skills'][0]
                nets[lr][st]['report'] = nets[lr][st]['net']['skills'][1]
                nets[lr][st]['cm'] = nets[lr][st]['net']['skills'][2]
                print ' => OK'
            except:
                try:
                    nets[lr][st]['net'] = load_net('../cache/trained/kitt_amter_nn_00_40_alls_allt_500&' + str(lr) + '_300_[' + str(st) + ']_1.net')
                    nets[lr][st]['acc_list'] = nets[lr][st]['net']['training_eval'][0]['v']
                    nets[lr][st]['err_list'] = nets[lr][st]['net']['training_eval'][1]['v']
                    nets[lr][st]['time_list'] = nets[lr][st]['net']['training_eval'][2]
                    nets[lr][st]['acc'] = nets[lr][st]['net']['skills'][0]
                    nets[lr][st]['report'] = nets[lr][st]['net']['skills'][1]
                    nets[lr][st]['cm'] = nets[lr][st]['net']['skills'][2]
                    print ' => OK'
                except:
                    nets[lr][st]['acc_list'] = [-1]*500
                    nets[lr][st]['err_list'] = [-1]*500
                    nets[lr][st]['time_list'] = [-1]*500
                    nets[lr][st]['acc'] = 0
                    nets[lr][st]['report'] = None
                    nets[lr][st]['cm'] = None
                    print ' => Failed'

    return nets


if __name__ == '__main__':
    nets = load()

    n_epochs = 500
    epochs = range(n_epochs+1)
    colors = ('red', 'green', 'blue', 'magenta', 'lime', 'cyan', 'orange', 'violet', 'brown', 'yellow', 'maroon', 'black')

    ''' Grid Search Analysis to find optimal learning params '''

    # Learning rate analysis
    plt.figure()
    st = 20
    for color, lr in zip(colors, sorted(nets.keys())):
        plt.plot(epochs, nets[lr][st]['acc_list'], '--', color=color, label=str(lr))
    plt.xlabel('training epoch')
    plt.ylabel('classification accuracy')
    plt.xlim([-1, len(epochs) + 1])
    plt.ylim([0, 1.1])
    plt.grid()
    plt.legend(loc='best')
    plt.show()

    # Network structure analysis
    plt.figure()
    lr = 0.1
    for color, st in zip(colors, sorted(nets[lr].keys())):
        plt.plot(epochs, nets[lr][st]['acc_list'], '--', color=color, label='[960, '+str(st)+', 14]')
    plt.xlabel('training epoch')
    plt.ylabel('classification accuracy')
    plt.xlim([-1, len(epochs) + 1])
    plt.ylim([0, 1.1])
    plt.grid()
    plt.legend(loc='best')
    plt.show()

    # Network structure vs Learning rate Final accuracy
    mat = list()
    lrs = sorted(nets.keys())
    sts = sorted(nets[0.1].keys())
    for st in sts:
        mat.append([nets[lr][st]['acc'] for lr in lrs])
    plt.matshow(mat, vmin=0.0, vmax=1.0)
    plt.xticks(range(len(lrs)), lrs)
    plt.yticks(range(len(sts)), ['[960, '+str(st)+', 14]' for st in sts])
    plt.xlabel('learning rate')
    plt.ylabel('network structure')
    plt.colorbar()

    for lr_i, lr in enumerate(lrs):
        for st_i, st in enumerate(sts):
            plt.text(lr_i, st_i, round(mat[st_i][lr_i], 2), va='center', ha='center', fontsize=11)
    plt.show()