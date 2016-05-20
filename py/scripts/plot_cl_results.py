#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from functions import load_net


def load():
    nets = dict()
    for tn in terrain_noise_types:
        nets[tn] = dict()
        for sn in signal_noise_types:
            sn_str = '0'+str(sn)[2:]
            nets[tn][sn] = dict()
            for ts in timesteps:
                nets[tn][sn][ts] = dict()
                for sen in sensors:
                    nets[tn][sn][ts][sen] = dict()
                    net_name = 'kitt_amter_' + tn + '_' + sn_str + '_' + str(ts) + '_' + sen + '_allt_500&' + str(
                        lr) + '_' + str(n_epochs) + '_[' + str(st) + ']'
                    nets[tn][sn][ts][sen]['net_name'] = net_name
                    nets[tn][sn][ts][sen]['has_data'] = True

                    for obs in range(1, n_obs+1):
                        nets[tn][sn][ts][sen][obs] = dict()
                        try:
                            nets[tn][sn][ts][sen][obs]['net'] = load_net('../cache/trained/'+net_name+'_'+str(obs)+'.net')

                            nets[tn][sn][ts][sen][obs]['acc_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][0]['v']
                            nets[tn][sn][ts][sen][obs]['err_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][1]['v']
                            nets[tn][sn][ts][sen][obs]['time_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][2]
                            nets[tn][sn][ts][sen][obs]['acc'] = nets[tn][sn][ts][sen][obs]['net']['skills'][0]
                            nets[tn][sn][ts][sen][obs]['report'] = nets[tn][sn][ts][sen][obs]['net']['skills'][1]
                            nets[tn][sn][ts][sen][obs]['cm'] = nets[tn][sn][ts][sen][obs]['net']['skills'][2]
                            #print ' => OK'
                        except:
                            nets[tn][sn][ts][sen][obs]['acc_list'] = [-1]*n_epochs
                            nets[tn][sn][ts][sen][obs]['err_list'] = [-1]*n_epochs
                            nets[tn][sn][ts][sen][obs]['time_list'] = [-1]*n_epochs
                            nets[tn][sn][ts][sen][obs]['acc'] = 0
                            nets[tn][sn][ts][sen][obs]['report'] = None
                            nets[tn][sn][ts][sen][obs]['cm'] = None
                            #print ' => Failed'
                            nets[tn][sn][ts][sen]['has_data'] = False

                    nets[tn][sn][ts][sen]['mean_acc'] = np.mean([nets[tn][sn][ts][sen][obs]['acc'] for obs in range(1, n_obs+1)])
                    nets[tn][sn][ts][sen]['mean_acc_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['acc_list'] for obs in range(1, n_obs + 1)], axis=0)
                    nets[tn][sn][ts][sen]['mean_err_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['err_list'] for obs in range(1, n_obs + 1)], axis=0)
                    nets[tn][sn][ts][sen]['mean_time_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['time_list'] for obs in range(1, n_obs + 1)], axis=0)

    return nets


if __name__ == '__main__':
    terrain_noise_types = ('nn', 'n1p', 'n3p', 'n5p', 'n10p', 'n20p')
    signal_noise_types = (0.0, 0.01, 0.03, 0.05, 0.1)
    timesteps = (1, 10, 40, 80)
    sensors = ('alls', 'angle', 'foot')
    n_epochs = 500
    lr = 0.5
    st = 20
    n_obs = 1

    nets = load()

    ''' Print results '''
    for tn in terrain_noise_types:
        for sn in signal_noise_types:
            for ts in timesteps:
                for sen in sensors:
                    if nets[tn][sn][ts][sen]['has_data']:
                        print '\n ## NET:', nets[tn][sn][ts][sen]['net_name']
                        print '@terrain noise:\t\t', tn
                        print '@signal noise:\t\t', sn
                        print '@timesteps:\t\t', ts
                        print '@sensors:\t\t', sen
                        print '@terrains:\t\t', 'allt'
                        print '@samples:\t\t', 500
                        print '@learning rate:\t\t', lr
                        print '@hidden neurons:\t', st
                        print '@epochs:\t\t', n_epochs
                        print '@jobs:\t\t\t', n_obs
                        print '---- # results:'
                        print '---- @accuracy:\t\t', nets[tn][sn][ts][sen]['mean_acc']
                        print '---- @precision:\t', nets[tn][sn][ts][sen]['mean_acc']
                        print '---- @recall:\t\t', nets[tn][sn][ts][sen]['mean_acc']
                        print '---- @f1-score:\t\t', nets[tn][sn][ts][sen]['mean_acc']
                        print '---- @error:\t\t', nets[tn][sn][ts][sen]['mean_err_list'][-1]
                        print '---- @avg. time:\t', nets[tn][sn][ts][sen]['mean_time_list'][-1]

    exit()
    epochs = range(n_epochs+1)
    colors = ('red', 'green', 'blue', 'magenta', 'yellow', 'cyan', 'black', 'orange', 'violet', 'brown', 'lime', 'maroon')

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