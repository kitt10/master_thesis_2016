#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 18

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
            these_obs = list()
            for obs in observations:
                nets[lr][st][obs] = dict()
                try:
                    nets[lr][st][obs]['net'] = load_net('../cache/trained/kitt_amter_nn_00_40_alls_allt_500&'+str(lr)+'_500_['+str(st)+']_'+str(obs)+'.net')

                    nets[lr][st][obs]['acc_list'] = nets[lr][st][obs]['net']['training_eval'][0]['v']
                    nets[lr][st][obs]['err_list'] = nets[lr][st][obs]['net']['training_eval'][1]['v']
                    nets[lr][st][obs]['time_list'] = nets[lr][st][obs]['net']['training_eval'][2]
                    nets[lr][st][obs]['acc'] = nets[lr][st][obs]['net']['skills'][0]
                    nets[lr][st][obs]['report'] = nets[lr][st][obs]['net']['skills'][1]
                    nets[lr][st][obs]['cm'] = nets[lr][st][obs]['net']['skills'][2]
                    print ' => OK'
                    these_obs.append(obs)
                except:
                    try:
                        nets[lr][st][obs]['net'] = load_net('../cache/trained/kitt_amter_nn_00_40_alls_allt_500&' + str(lr) + '_300_[' + str(st) + ']_'+str(obs)+'.net')
                        nets[lr][st][obs]['acc_list'] = nets[lr][st][obs]['net']['training_eval'][0]['v']
                        nets[lr][st][obs]['err_list'] = nets[lr][st][obs]['net']['training_eval'][1]['v']
                        nets[lr][st][obs]['time_list'] = nets[lr][st][obs]['net']['training_eval'][2]
                        nets[lr][st][obs]['acc'] = nets[lr][st][obs]['net']['skills'][0]
                        nets[lr][st][obs]['report'] = nets[lr][st][obs]['net']['skills'][1]
                        nets[lr][st][obs]['cm'] = nets[lr][st][obs]['net']['skills'][2]
                        print ' => OK'
                        these_obs.append(obs)
                    except:
                        nets[lr][st]['acc_list'] = [-1]*500
                        nets[lr][st]['err_list'] = [-1]*500
                        nets[lr][st]['time_list'] = [-1]*500
                        nets[lr][st]['acc'] = 0
                        nets[lr][st]['report'] = None
                        nets[lr][st]['cm'] = None
                        print ' => Failed'

            nets[lr][st]['mean_acc'] = np.mean([nets[lr][st][obs]['acc'] for obs in these_obs])
            nets[lr][st]['std_acc'] = np.std([nets[lr][st][obs]['acc'] for obs in these_obs])
            nets[lr][st]['mean_acc_list'] = np.mean([nets[lr][st][obs]['acc_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['std_acc_list'] = np.std([nets[lr][st][obs]['acc_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['mean_err_list'] = np.mean([nets[lr][st][obs]['err_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['std_err_list'] = np.std([nets[lr][st][obs]['err_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['mean_time_list'] = np.mean([nets[lr][st][obs]['time_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['std_time_list'] = np.std([nets[lr][st][obs]['time_list'][:n_epochs] for obs in these_obs], axis=0)
            nets[lr][st]['mean_time'] = np.mean(nets[lr][st]['mean_time_list'])

    return nets


if __name__ == '__main__':

    n_epochs = 300
    n_obs = 5
    observations = range(1, n_obs+1)
    epochs = range(1, n_epochs+1)
    colors = ('red', 'green', 'maroon', 'magenta', 'lime', 'cyan', 'black', 'yellow', 'blue', 'violet', 'brown', 'black')

    nets = load()

    ''' Grid Search Analysis to find optimal learning params '''

    # Learning rate analysis
    plt.figure()
    st = 20
    colors=('red', 'green', 'blue', 'black', 'gold')
    #for color, lr in zip(colors, sorted(nets.keys())):
    for color, lr in zip(colors, (0.01, 0.05, 0.1, 0.5, 0.9)):
        #plt.plot(epochs, nets[lr][st]['mean_acc_list'], '-', color=color, label=str(lr))
        plt.errorbar(x=epochs, y=nets[lr][st]['mean_acc_list'], yerr=nets[lr][st]['std_acc_list'], color=color, label=str(lr))
    plt.xlabel('training epoch')
    plt.ylabel('classification accuracy')
    plt.xlim([-1, len(epochs) + 1])
    plt.ylim([0, 1])
    plt.grid()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='best')
    plt.show()

    # Network structure analysis
    plt.figure()
    lr = 0.5
    colors = ('red', 'green', 'blue', 'black', 'gold', 'magenta')
    #for color, st in zip(colors, sorted(nets[lr].keys())):
    for color, st in zip(colors, (5, 7, 10, 20, 50, 100)):
        #plt.plot(epochs, nets[lr][st]['mean_acc_list'], '-', color=color, label=str(st))
        plt.errorbar(x=epochs, y=nets[lr][st]['mean_acc_list'], yerr=nets[lr][st]['std_acc_list'], color=color, label=st)
    plt.xlabel('training epoch')
    plt.ylabel('classification accuracy')
    plt.xlim([-1, len(epochs) + 1])
    plt.ylim([0, 1])
    plt.grid()

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='best')
    plt.show()

    # Network structure vs Learning rate Final accuracy
    mat = list()
    lrs = sorted(nets.keys())
    sts = sorted(nets[0.1].keys())
    for st in sts:
        mat.append([nets[lr][st]['mean_acc'] for lr in lrs])
    plt.matshow(mat, vmin=0.0, vmax=1.0)
    plt.xticks(range(len(lrs)), lrs, rotation=30)
    plt.yticks(range(len(sts)), ['[960, '+str(st)+', 14]' for st in sts])
    plt.xlabel('learning rate')
    plt.ylabel('network structure')
    plt.colorbar()

    for lr_i, lr in enumerate(lrs):
        for st_i, st in enumerate(sts):
            plt.text(lr_i, st_i, round(mat[st_i][lr_i], 2), va='center', ha='center', fontsize=11)
    #plt.savefig('../../thesis/img/cl_st_lr_mat.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()