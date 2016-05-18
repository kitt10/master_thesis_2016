#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from functions import load_net


def load_nets():
    kitt_xor = {'acc': list(), 'n_syn': list(), 'structure': list(), 'acc_mean': None, 'n_syn_mean': None,
                'acc_std': None, 'n_syn_std': None, 'structure_mean': None}
    n_obs = 10
    max_len = 0
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_xor_to_prune_p_' + str(obs) + '.net')
        if len(net['pruning_eval'][0]) > max_len:
            max_len = len(net['pruning_eval'][0])
    for obs in range(1, n_obs):
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

if __name__ == '__main__':
    kitt_xor = load_nets()

    ''' XOR PA result '''
    fig, ax1 = plt.subplots()
    pruning_steps = range(len(kitt_xor['acc_mean']))
    ax1.errorbar(x=pruning_steps, y=kitt_xor['acc_mean'], yerr=kitt_xor['acc_std'], color='darkgreen')
    ax1.set_xlabel('pruning step')
    ax1.set_ylabel('classification accuracy', color='darkgreen')
    ax1.set_ylim([0, 1.1])
    for tl in ax1.get_yticklabels():
        tl.set_color('darkgreen')

    ax2 = ax1.twinx()
    ax2.errorbar(x=range(len(kitt_xor['n_syn_mean'])), y=kitt_xor['n_syn_mean'], yerr=kitt_xor['n_syn_std'], color='maroon')
    ax2.set_ylabel('number of synapses', color='maroon')
    ax2.set_ylim([0, 450])
    for tl in ax2.get_yticklabels():
        tl.set_color('maroon')

    blue_patch = mpatches.Patch(color='darkblue', label='Net structure')
    plt.legend([blue_patch], [p.get_label() for p in [blue_patch]], loc='center right')

    # Annotate
    plt.annotate('[2. 100. 2]', xy=(0, 420), horizontalalignment='center', verticalalignment='center',
                 fontsize=11, color='darkblue')
    for step in pruning_steps:
        if step % 5 == 0 or step < 3:
            plt.annotate(str(kitt_xor['structure_mean'][step]), xy=(step+0.5, kitt_xor['n_syn_mean'][step] + 0.7 * kitt_xor['n_syn_mean'][step]),
                         horizontalalignment='center', verticalalignment='center', fontsize=11, color='darkblue')

    plt.grid()
    plt.xlim([-1, len(pruning_steps)+1])
    #plt.savefig('../../thesis/img/pa_result_xor.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()