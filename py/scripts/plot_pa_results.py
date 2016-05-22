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


def load_xor():
    kitt_xor = {'acc': list(), 'n_syn': list(), 'structure': list(), 'acc_mean': None, 'n_syn_mean': None,
                'acc_std': None, 'n_syn_std': None, 'structure_mean': None}
    n_obs = 10
    max_len = 0
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_xor_to_prune_p_' + str(obs) + '.net')
        if len(net['pruning_eval'][0]) > max_len:
            max_len = len(net['pruning_eval'][0])
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_xor_to_prune_p_' + str(obs) + '.net')
        acc_list = net['pruning_eval'][0]
        while len(acc_list) < max_len:
            acc_list.append(acc_list[-1])
        kitt_xor['acc'].append(acc_list)
        n_syn_list = net['pruning_eval'][1]
        while len(n_syn_list) < max_len:
            n_syn_list.append(n_syn_list[-1])
        kitt_xor['n_syn'].append(n_syn_list)
        structure_list = net['pruning_eval'][2]
        while len(structure_list) < max_len:
            structure_list.append(structure_list[-1])
        kitt_xor['structure'].append(structure_list)

    kitt_xor['acc_mean'] = np.mean(kitt_xor['acc'], axis=0)
    kitt_xor['acc_std'] = np.std(kitt_xor['acc'], axis=0)
    kitt_xor['n_syn_mean'] = np.mean(kitt_xor['n_syn'], axis=0)
    kitt_xor['n_syn_std'] = np.std(kitt_xor['n_syn'], axis=0)
    kitt_xor['structure_mean'] = np.mean([structure for structure in kitt_xor['structure']], axis=0)

    for i, structure in enumerate(kitt_xor['structure_mean']):
        kitt_xor['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded XOR pruned nets ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_xor['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_xor['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_xor['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_xor


def load_mnist():
    kitt_xor = {'acc': list(), 'n_syn': list(), 'structure': list(), 'acc_mean': None, 'n_syn_mean': None,
                'acc_std': None, 'n_syn_std': None, 'structure_mean': None}
    n_obs = 3
    max_len = 0
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_mnist_to_prune_p_89_' + str(obs) + '.net')
        if len(net['pruning_eval'][0]) > max_len:
            max_len = len(net['pruning_eval'][0])

    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_mnist_to_prune_p_89_' + str(obs) + '.net')
        acc_list = net['pruning_eval'][0]
        while len(acc_list) < max_len:
            acc_list.append(acc_list[-1])
        kitt_xor['acc'].append(acc_list)
        n_syn_list = net['pruning_eval'][1]
        while len(n_syn_list) < max_len:
            n_syn_list.append(n_syn_list[-1])
        kitt_xor['n_syn'].append(n_syn_list)
        structure_list = net['pruning_eval'][2]
        while len(structure_list) < max_len:
            structure_list.append(structure_list[-1])
        kitt_xor['structure'].append(structure_list)

    kitt_xor['acc_mean'] = np.mean(kitt_xor['acc'], axis=0)
    kitt_xor['acc_std'] = np.std(kitt_xor['acc'], axis=0)
    kitt_xor['n_syn_mean'] = np.mean(kitt_xor['n_syn'], axis=0)
    kitt_xor['n_syn_std'] = np.std(kitt_xor['n_syn'], axis=0)
    kitt_xor['structure_mean'] = np.mean([structure for structure in kitt_xor['structure']], axis=0)

    for i, structure in enumerate(kitt_xor['structure_mean']):
        kitt_xor['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded XOR pruned nets ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_xor['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_xor['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_xor['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_xor


def load_amter(na):
    kitt_amter = {'acc': list(), 'n_syn': list(), 'structure': list(), 'acc_mean': None, 'n_syn_mean': None,
                'acc_std': None, 'n_syn_std': None, 'structure_mean': None}
    n_obs = 5
    max_len = 0
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_amter_to_prune_'+na+'_p_' + str(obs) + '.net')
        if len(net['pruning_eval'][0]) > max_len:
            max_len = len(net['pruning_eval'][0])

    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_amter_to_prune_'+na+'_p_' + str(obs) + '.net')
        acc_list = net['pruning_eval'][0]
        while len(acc_list) < max_len:
            acc_list.append(acc_list[-1])
        kitt_amter['acc'].append(acc_list)
        n_syn_list = net['pruning_eval'][1]
        while len(n_syn_list) < max_len:
            n_syn_list.append(n_syn_list[-1])
        kitt_amter['n_syn'].append(n_syn_list)
        structure_list = net['pruning_eval'][2]
        while len(structure_list) < max_len:
            structure_list.append(structure_list[-1])
        kitt_amter['structure'].append(structure_list)

    kitt_amter['acc_mean'] = np.mean(kitt_amter['acc'], axis=0)
    kitt_amter['acc_std'] = np.std(kitt_amter['acc'], axis=0)
    kitt_amter['n_syn_mean'] = np.mean(kitt_amter['n_syn'], axis=0)
    kitt_amter['n_syn_std'] = np.std(kitt_amter['n_syn'], axis=0)
    kitt_amter['structure_mean'] = np.mean([structure for structure in kitt_amter['structure']], axis=0)

    for i, structure in enumerate(kitt_amter['structure_mean']):
        kitt_amter['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded XOR pruned nets ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_amter['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_amter['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_amter['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_amter

if __name__ == '__main__':
    kitt_net = load_amter(na='noisy')

    ''' XOR PA result '''
    fig, ax1 = plt.subplots()
    pruning_steps = range(len(kitt_net['acc_mean']))
    ax1.errorbar(x=pruning_steps, y=kitt_net['acc_mean'], yerr=kitt_net['acc_std'], color='darkgreen')
    ax1.set_xlabel('pruning step')
    ax1.set_ylabel('classification accuracy', color='darkgreen')
    ax1.set_ylim([0, 1.1])
    for tl in ax1.get_yticklabels():
        tl.set_color('darkgreen')

    ax2 = ax1.twinx()
    ax2.errorbar(x=range(len(kitt_net['n_syn_mean'])), y=kitt_net['n_syn_mean'], yerr=kitt_net['n_syn_std'], color='maroon')
    ax2.set_ylabel('number of synapses', color='maroon')
    ax2.set_ylim([0, 20000])
    for tl in ax2.get_yticklabels():
        tl.set_color('maroon')

    blue_patch = mpatches.Patch(color='darkblue', label='Net structure')
    plt.legend([blue_patch], [p.get_label() for p in [blue_patch]], loc='center right')

    # Annotate
    '''
    plt.annotate('[960.20.14.]', xy=(0, 19950), horizontalalignment='center', verticalalignment='center',
                 fontsize=11, color='darkblue')
    '''

    for step in pruning_steps:
        if step % 25 == 0 or step < 3:
            plt.annotate(str(kitt_net['structure_mean'][step]).replace(' ', ''), xy=(step+11.5, kitt_net['n_syn_mean'][step] + 0.9 * kitt_net['n_syn_mean'][step]),
                         horizontalalignment='center', verticalalignment='center', fontsize=13, color='darkblue')

    plt.grid()
    plt.xlim([-1, len(pruning_steps)+1])
    #plt.savefig('../../thesis/img/pa_result_mnist.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()