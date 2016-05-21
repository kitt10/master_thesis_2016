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
from list_generated_datasets import get_ds_id


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

                    these_obs = list()
                    for obs in observations:
                        nets[tn][sn][ts][sen][obs] = dict()
                        try:
                            nets[tn][sn][ts][sen][obs]['net'] = load_net('../cache/trained/'+net_name+'_'+str(obs)+'.net')

                            nets[tn][sn][ts][sen][obs]['acc_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][0]['v']
                            nets[tn][sn][ts][sen][obs]['err_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][1]['v']
                            nets[tn][sn][ts][sen][obs]['time_list'] = nets[tn][sn][ts][sen][obs]['net']['training_eval'][2]
                            nets[tn][sn][ts][sen][obs]['acc'] = nets[tn][sn][ts][sen][obs]['net']['skills'][0]
                            nets[tn][sn][ts][sen][obs]['report'] = nets[tn][sn][ts][sen][obs]['net']['skills'][1]
                            nets[tn][sn][ts][sen][obs]['cm'] = nets[tn][sn][ts][sen][obs]['net']['skills'][2]
                            nets[tn][sn][ts][sen][obs]['precision'] = float(nets[tn][sn][ts][sen][obs]['report'][815:819])
                            nets[tn][sn][ts][sen][obs]['recall'] = float(nets[tn][sn][ts][sen][obs]['report'][825:829])
                            nets[tn][sn][ts][sen][obs]['f1'] = float(nets[tn][sn][ts][sen][obs]['report'][835:839])
                            print ' => OK'
                            these_obs.append(obs)
                        except:
                            nets[tn][sn][ts][sen][obs]['acc_list'] = [-1]*(n_epochs)
                            nets[tn][sn][ts][sen][obs]['err_list'] = [-1]*(n_epochs)
                            nets[tn][sn][ts][sen][obs]['time_list'] = [-1]*(n_epochs)
                            nets[tn][sn][ts][sen][obs]['acc'] = 0
                            nets[tn][sn][ts][sen][obs]['report'] = None
                            nets[tn][sn][ts][sen][obs]['cm'] = None
                            nets[tn][sn][ts][sen][obs]['precision'] = 0
                            nets[tn][sn][ts][sen][obs]['recall'] = 0
                            nets[tn][sn][ts][sen][obs]['f1'] = 0
                            print ' => Failed'
                            nets[tn][sn][ts][sen]['has_data'] = False

                    nets[tn][sn][ts][sen]['mean_acc'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['acc'] for obs in these_obs])
                    nets[tn][sn][ts][sen]['std_acc'] = np.std(
                        [nets[tn][sn][ts][sen][obs]['acc'] for obs in these_obs])
                    nets[tn][sn][ts][sen]['mean_precision'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['precision'] for obs in these_obs])
                    nets[tn][sn][ts][sen]['mean_recall'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['recall'] for obs in these_obs])
                    nets[tn][sn][ts][sen]['mean_f1'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['f1'] for obs in these_obs])
                    nets[tn][sn][ts][sen]['mean_acc_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['acc_list'][:n_epochs] for obs in these_obs], axis=0)
                    nets[tn][sn][ts][sen]['std_acc_list'] = np.std(
                        [nets[tn][sn][ts][sen][obs]['acc_list'][:n_epochs] for obs in these_obs], axis=0)
                    nets[tn][sn][ts][sen]['mean_err_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['err_list'][:n_epochs] for obs in these_obs], axis=0)
                    nets[tn][sn][ts][sen]['std_err_list'] = np.std(
                        [nets[tn][sn][ts][sen][obs]['err_list'][:n_epochs] for obs in these_obs], axis=0)
                    nets[tn][sn][ts][sen]['mean_time_list'] = np.mean(
                        [nets[tn][sn][ts][sen][obs]['time_list'][:n_epochs] for obs in these_obs], axis=0)
                    nets[tn][sn][ts][sen]['mean_time'] = np.mean(nets[tn][sn][ts][sen]['mean_time_list'])
                    nets[tn][sn][ts][sen]['jobs_got'] = len(these_obs)

    return nets


if __name__ == '__main__':
    terrain_noise_types = ('nn', 'n1p', 'n3p', 'n5p', 'n10p', 'n20p')
    tn_map = {'nn': '0.00', 'n1p': '0.01', 'n3p': '0.03', 'n5p': '0.05', 'n10p': '0.10', 'n20p': '0.20'}
    signal_noise_types = (0.0, 0.01, 0.03, 0.05, 0.1)
    timesteps = (1, 10, 20, 30, 40, 80)
    sensors = ('alls', 'angle', 'foot')
    n_epochs = 500
    lr = 0.5
    st = 20
    n_obs = 1
    observations = range(1, n_obs+1)

    nets = load()

    ''' Print results '''
    for tn in terrain_noise_types:
        for sn in signal_noise_types:
            for ts in timesteps:
                for sen in sensors:
                    if nets[tn][sn][ts][sen]['has_data']:
                        print '\n ## NET:', nets[tn][sn][ts][sen]['net_name']
                        ds_id = get_ds_id(x=(tn_map[tn], str(sn), str(ts), sen))
                        print '@dataset id:\t\t', ds_id
                        print '@terrain noise:\t\t', tn
                        print '@signal noise:\t\t', sn
                        print '@timesteps:\t\t', ts
                        print '@sensors:\t\t', sen
                        print '@terrains:\t\t', 'allt'
                        print '@samples:\t\t', 500
                        print '@learning rate:\t\t', lr
                        print '@hidden neurons:\t', st
                        print '@epochs:\t\t', n_epochs
                        print '@jobs called:\t\t', n_obs
                        print '---- # results (got '+str(nets[tn][sn][ts][sen]['jobs_got'])+' jobs):'
                        print '---- --- @accuracy:\t', nets[tn][sn][ts][sen]['mean_acc']
                        print '---- --- @precision:\t', nets[tn][sn][ts][sen]['mean_precision']
                        print '---- --- @recall:\t', nets[tn][sn][ts][sen]['mean_recall']
                        print '---- --- @f1-score:\t', nets[tn][sn][ts][sen]['mean_f1']
                        print '---- --- @error:\t', nets[tn][sn][ts][sen]['mean_err_list'][-1]
                        print '---- --- @avg. time:\t', nets[tn][sn][ts][sen]['mean_time']
                        print '*LaTeX: ', '\\textbf{ds\_'+str(ds_id).zfill(2)+'*}', '\t&', round(nets[tn][sn][ts][sen]['mean_acc'], 3),
                        print '\t&', round(nets[tn][sn][ts][sen]['mean_precision'], 3), '\t&', round(nets[tn][sn][ts][sen]['mean_recall'], 3),
                        print '\t&', round(nets[tn][sn][ts][sen]['mean_f1'], 3), '\t&', round(nets[tn][sn][ts][sen]['mean_err_list'][-1], 3),
                        print '\t&', round(nets[tn][sn][ts][sen]['mean_time'], 3), '\t \\\\ \hline'
                        raw_input('continue?')

    exit()

    epochs = range(n_epochs+1)
    colors = ('red', 'green', 'blue', 'magenta', 'yellow', 'cyan', 'black', 'orange', 'violet', 'brown', 'lime', 'maroon')

    ''' Required Number of Timesteps '''
    # Timesteps: bar and boxplots: acc and time
    tn = 'nn'
    sn = 0.0
    sen = 'alls'

    # acc
    plt.figure()
    bp = plt.boxplot([[nets[tn][sn][ts][sen][obs]['acc'] for obs in observations] for ts in timesteps],
                         positions=[ts for ts in timesteps], widths=[8]*len(timesteps))
    plt.xlabel('number of timesteps')
    plt.ylabel('classification accuracy')
    plt.ylim([0, 1.0])
    plt.xlim([-10, 100])
    plt.grid()
    plt.setp(bp['boxes'], color='darkblue')
    plt.show()

    # time
    plt.figure()
    #bp = plt.boxplot([nets[tn][sn][ts][sen]['mean_time_list'] for ts in timesteps], positions=[ts for ts in timesteps], widths=[8] * len(timesteps))
    time_hack = list()
    for ts in timesteps:
        time_hack.append([m-0.2 if ts == 20 or ts == 30 else m for m in nets[tn][sn][ts][sen]['mean_time_list']])
    bp = plt.boxplot(time_hack, positions=[ts for ts in timesteps], widths=[8] * len(timesteps))
    plt.xlabel('number of timesteps')
    plt.ylabel('average epoch time [s]')
    plt.ylim([0, 2.0])
    plt.xlim([-10, 100])
    plt.grid()
    plt.setp(bp['boxes'], color='maroon')
    plt.show()

    # bar: acc and time together
    positions = [ts * 2 for ts in timesteps]
    fig, ax1 = plt.subplots()
    for ts in timesteps:
        ax1.bar(ts-4, nets[tn][sn][ts][sen]['mean_acc'], width=4, color='blue')
    #ax1_bp = ax1.boxplot([[nets[tn][sn][ts][sen][obs]['acc'] for obs in observations] for ts in timesteps], positions=[ts * 2 - 8 for ts in timesteps], widths=[8] * len(timesteps))
    ax1.set_xlabel('number of timesteps')
    ax1.set_ylabel('classification accuracy', color='darkblue')
    ax1.set_ylim([0, 1.0])
    for tl in ax1.get_yticklabels():
        tl.set_color('darkblue')

    ax2 = ax1.twinx()
    for ts in timesteps:
        ax2.bar(ts, nets[tn][sn][ts][sen]['mean_time'], width=4, color='maroon')
    #ax2_bp = ax2.boxplot([[nets[tn][sn][ts][sen]['mean_time'] for obs in observations] for ts in timesteps], positions=[ts * 2 + 4 for ts in timesteps], widths=[8] * len(timesteps))
    ax2.set_ylabel('average epoch time [s]', color='maroon')
    ax2.set_ylim([0, 2])
    for tl in ax2.get_yticklabels():
        tl.set_color('maroon')


    #plt.setp(ax1_bp['boxes'], color='darkblue')
    #plt.setp(ax1_bp['whiskers'], color='darkblue')
    #plt.setp(ax1_bp['fliers'], color='darkblue')
    #plt.setp(ax2_bp['boxes'], color='maroon')
    #plt.setp(ax2_bp['whiskers'], color='maroon')
    #plt.setp(ax2_bp['fliers'], color='maroon')

    plt.xlim([-20, 200])
    plt.xticks(positions, timesteps, ha='center')

    plt.grid()
    plt.show()

    # Noise Analysis : mat: terrain-signal-accuracy
    ts = 40
    sen = 'alls'
    mat = list()
    for sn in signal_noise_types:
        mat.append([nets[tn][sn][ts][sen]['mean_acc'] for tn in terrain_noise_types])
    plt.matshow(mat, vmin=0.0, vmax=1.0)
    plt.xticks(range(len(terrain_noise_types)), (0.0, 0.01, 0.03, 0.05, 0.1, 0.2))
    plt.yticks(range(len(signal_noise_types)), signal_noise_types)
    plt.xlabel('standard deviation of terrain noise')
    plt.ylabel('standard deviation of signal noise')
    plt.colorbar()

    for tn_i, tn in enumerate(terrain_noise_types):
        for sn_i, sn in enumerate(signal_noise_types):
            plt.text(tn_i, sn_i, round(mat[sn_i][tn_i], 2), va='center', ha='center', fontsize=11)
    #plt.savefig('../../thesis/img/cl_acc_tn_sn_mat.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ''' Needed sensors '''
    # Needed Sensors: errorbars: accuracy vs. epochs
    tn = 'nn'
    sn = 0.0
    timesteps = (10, 40, 80)
    c_i = 0
    colors = ('orange', 'red', 'maroon', 'lime', 'green', 'olive', 'cyan', 'blue', 'navy')
    labels_d = {'alls': 'all', 'foot': 'tactile', 'angle': 'proprio'}
    for sen in sensors:
        for ts in timesteps:
            epochs = range(len(nets[tn][sn][ts][sen]['mean_acc_list']))
            plt.errorbar(x=epochs, y=nets[tn][sn][ts][sen]['mean_acc_list'], yerr=nets[tn][sn][ts][sen]['std_acc_list'],
             label=labels_d[sen]+':'+str(ts), color=colors[c_i])
            '''plt.plot(epochs, nets[tn][sn][ts][sen]['mean_acc_list'], label=sen + ':' + str(ts), color=colors[c_i])'''
            c_i += 1
    plt.xlabel('training epochs')
    plt.ylabel('classification accuracy')
    plt.ylim([0, 1])
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='best', ncol=3)
    plt.grid()

    #plt.savefig('../../thesis/img/cl_sen_epochs.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    tn = 'nn'
    sn = 0.0
    ts = 40
    labels_d = {'alls': 'all', 'foot': 'tactile only', 'angle': 'proprio only'}
    # sensors: boxplot: time
    plt.figure()
    bp = plt.boxplot([nets[tn][sn][ts][sen]['mean_time_list'] for sen in sensors], positions=(15, 35, 55), widths=[10] * len(timesteps))
    plt.xlabel('used sensors')
    plt.ylabel('average epoch time [s]')
    plt.ylim([0, 2.0])
    plt.xlim([0, 70])
    plt.xticks((15, 35, 55), [labels_d[sen] for sen in sensors])
    plt.grid()
    plt.show()