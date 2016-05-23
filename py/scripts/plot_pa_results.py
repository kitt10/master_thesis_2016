#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 18

import matplotlib.pyplot as plt
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


def load_xor_comp(na):
    kitt_xor_comp = {'acc': list(), 'n_syn': list(), 'structure': list(), 'acc_mean': None, 'n_syn_mean': None,
                'acc_std': None, 'n_syn_std': None, 'structure_mean': None}
    n_obs = 10
    max_len = 0
    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_mnist_to_prune_'+na+'_p_' + str(obs) + '.net')
        if len(net['pruning_eval'][0]) > max_len:
            max_len = len(net['pruning_eval'][0])

    for obs in range(1, n_obs+1):
        net = load_net('../cache/pruned/kitt_xor_to_prune_'+na+'_p_' + str(obs) + '.net')
        acc_list = net['pruning_eval'][0]
        while len(acc_list) < max_len:
            acc_list.append(acc_list[-1])
        kitt_xor_comp['acc'].append(acc_list)
        n_syn_list = net['pruning_eval'][1]
        while len(n_syn_list) < max_len:
            n_syn_list.append(n_syn_list[-1])
        kitt_xor_comp['n_syn'].append(n_syn_list)
        structure_list = net['pruning_eval'][2]
        while len(structure_list) < max_len:
            structure_list.append(structure_list[-1])
        kitt_xor_comp['structure'].append(structure_list)

    kitt_xor_comp['acc_mean'] = np.mean(kitt_xor_comp['acc'], axis=0)
    kitt_xor_comp['acc_std'] = np.std(kitt_xor_comp['acc'], axis=0)
    kitt_xor_comp['n_syn_mean'] = np.mean(kitt_xor_comp['n_syn'], axis=0)
    kitt_xor_comp['n_syn_std'] = np.std(kitt_xor_comp['n_syn'], axis=0)
    kitt_xor_comp['structure_mean'] = np.mean([structure for structure in kitt_xor_comp['structure']], axis=0)

    for i, structure in enumerate(kitt_xor_comp['structure_mean']):
        kitt_xor_comp['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded XOR pruned nets ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_xor_comp['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_xor_comp['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_xor_comp['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_xor_comp


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


if __name__ == '__main__':
    #na = 'noisy'
    #kitt_net = load_amter(na=na)

    ''' PA result '''
    '''
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

    for step in pruning_steps:
        if step % 20 == 0 or step < 5:
            plt.annotate(str(kitt_net['structure_mean'][step]).replace(' ', ''), xy=(step+7.5, kitt_net['n_syn_mean'][step] + 1.7 * kitt_net['n_syn_mean'][step]),
                         horizontalalignment='center', verticalalignment='center', fontsize=13, color='darkblue')

    plt.grid()
    plt.xlim([-1, len(pruning_steps)+1])
    #plt.savefig('../../thesis/img/pa_result_amter_'+na+'.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    '''

    # Number of neurons in percentage
    '''
    net_names = ('nn', 'noisy', '80', 'st100')
    name_dict = {'nn': 'nn_40_20_a', 'noisy': 'noisy_40_20_a', '80': 'nn_80_20_a', 'st100': 'nn_40_100_a', 'angle': 'nn_40_20_p', 'foot': 'nn_40_20_t'}
    #net_names = ('nn', 'angle', 'foot')
    inits = {'nn': (960, 20, 14), 'noisy': (960, 20, 14), '80': (1920, 20, 14), 'st100': (960, 100, 14),
             'foot': (240, 20, 14), 'angle': (720, 20, 14)}
    accs = {'nn': 0.9, 'noisy': 0.7, '80': 0.9, 'st100': 0.75,
             'foot': 0.65, 'angle': 0.65}
    nets = list()
    for net_name in net_names:
        nets.append(load_amter(na=net_name))
    for n_i, (net, net_name) in enumerate(zip(nets, net_names)):
        plt.bar(11+n_i*40, width=28, height=120, color='whitesmoke', alpha=0.2)
        plt.boxplot([[100.0*structure[-1][l_i]/inits[net_name][l_i] for structure in net['structure']] for l_i in range(3)],
                    positions=[15+n_i*40, 25+n_i*40, 35+n_i*40], widths=[8]*3)
        plt.annotate(name_dict[net_name], xy=(25 + n_i * 40, 120), va='center', ha='center', backgroundcolor='moccasin',
                     fontsize=15)
        plt.annotate('acc: '+str(accs[net_name]), xy=(25 + n_i * 40, 112), va='center', ha='center', color='green',
                     fontsize=13)
        for l_i in range(3):
            plt.annotate(inits[net_name][l_i], xy=(15+l_i*10 + n_i * 40, 105), va='center', ha='center', fontsize=13)
    plt.xlabel('networks/layers')
    plt.ylabel('used neurons after pruning [%]')
    plt.ylim([0, 125])
    plt.xlim([0, 50+40*(len(net_names)-1)])
    plt.xticks([15, 25, 35, 55, 65, 75, 95, 105, 115, 135, 145, 155], ['I', 'H', 'O', 'I', 'H', 'O', 'I', 'H', 'O', 'I', 'H', 'O'])
    plt.grid()
    plt.show()


    # synapses reduction over pruning steps

    #net_names = ('nn', 'noisy', '80', 'st100')
    name_dict = {'nn': 'nn_40_20_a', 'noisy': 'noisy_40_20_a', '80': 'nn_80_20_a', 'st100': 'nn_40_100_a',
                 'angle': 'nn_40_20_p', 'foot': 'nn_40_20_t'}
    name_dict2 = {'nn': 'all sensors (acc: 0.9)', 'angle': 'proprioceptive only (acc: 0.65)', 'foot': 'tactile only (acc: 0.65)'}
    net_names = ('nn', 'angle', 'foot')
    inits = {'nn': 1948, 'noisy': 1948, '80': 38680, 'st100': 97400,
             'foot': 5080, 'angle': 14680}
    accs = {'nn': 0.9, 'noisy': 0.7, '80': 0.9, 'st100': 0.75,
            'foot': 0.65, 'angle': 0.65}
    nets = list()
    for net_name in net_names:
        nets.append(load_amter(na=net_name))
    colors = ('red', 'green', 'blue', 'black', 'magenta', 'yellow', 'orange', 'brown')
    for n_i, (net, net_name) in enumerate(zip(nets, net_names)):
        #plt.errorbar(x=range(len(net['n_syn_mean'])), y=net['n_syn_mean'], yerr=net['n_syn_std'], color=colors[n_i], label=name_dict[net_name])
        errorfill(x=range(len(net['n_syn_mean'])), y=net['n_syn_mean'], yerr=net['n_syn_std'], color=colors[n_i], label=name_dict2[net_name])
    plt.xlabel('pruning step')
    plt.ylabel('number of synapses')
    plt.xlim([0, 200])
    plt.ylim([0, 1000])
    plt.grid()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    #handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='best')
    plt.show()

    '''

    # comparison to other pruning steps, synapses reduction over pruning step

    name_dict = {'kitt': 'weight changes', 'zero': 'weights close to zero',
                  'reed': 'brute force'}
    net_names = ('kitt', 'zero', 'reed')
    inits = {'kitt': 300, 'zero': 300, 'reed': 300}
    accs = {'kitt': 0.99, 'zero': 0.99, 'reed': 0.99}
    nets = list()
    for net_name in net_names:
        nets.append(load_xor_comp(na=net_name))
    colors = ('red', 'green', 'blue', 'black', 'magenta', 'yellow', 'orange', 'brown')
    for n_i, (net, net_name) in enumerate(zip(nets, net_names)):
        errorfill(x=range(len(net['n_syn_mean'])), y=net['n_syn_mean'], yerr=net['n_syn_std'], color=colors[n_i],
                  label=name_dict[net_name])
    plt.xlabel('pruning step')
    plt.ylabel('number of synapses')
    plt.xlim([0, 200])
    plt.ylim([0, 400])
    plt.grid()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # handles = [h[0] for h in handles]
    plt.legend(handles, labels, loc='best')
    plt.show()