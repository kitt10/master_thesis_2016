#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 18

import matplotlib.pyplot as plt
import numpy as np
from random import choice
from PIL import Image
from functions import load_net, norm_signal, add_signal_noise, load_params, read_data


def prepare_signal(signal, sen):
    """
    :param signal: raw signal
    :param sen: sensor used to measure this signal
    :return: normalized signal with a signal noise
    """
    global norm, signal_noise_std

    ''' First, normalize the signal '''
    normed_signal = norm_signal(signal=signal, the_min=sensors_ranges[sen][0], the_max=sensors_ranges[sen][1])

    ''' Adding signal noise of defined std '''
    noised_signal = add_signal_noise(signal=normed_signal, std=0)
    return noised_signal

def load_amter(na):
    kitt_amter = {'acc': list(), 'n_syn': list(), 'structure': list(), 'net': list(), 'acc_mean': None, 'n_syn_mean': None,
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
        kitt_amter['net'].append(net['net'])

    kitt_amter['acc_mean'] = np.mean(kitt_amter['acc'], axis=0)
    kitt_amter['acc_std'] = np.std(kitt_amter['acc'], axis=0)
    kitt_amter['n_syn_mean'] = np.mean(kitt_amter['n_syn'], axis=0)
    kitt_amter['n_syn_std'] = np.std(kitt_amter['n_syn'], axis=0)
    kitt_amter['structure_mean'] = np.mean([structure for structure in kitt_amter['structure']], axis=0)

    for i, structure in enumerate(kitt_amter['structure_mean']):
        kitt_amter['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded AMTER pruned nets ('+str(n_obs)+' obs) ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_amter['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_amter['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_amter['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_amter


def load_mnist():
    kitt_mnist = {'acc': list(), 'n_syn': list(), 'structure': list(), 'net': list(), 'acc_mean': None, 'n_syn_mean': None,
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
        kitt_mnist['acc'].append(acc_list)
        n_syn_list = net['pruning_eval'][1]
        while len(n_syn_list) < max_len:
            n_syn_list.append(n_syn_list[-1])
        kitt_mnist['n_syn'].append(n_syn_list)
        structure_list = net['pruning_eval'][2]
        while len(structure_list) < max_len:
            structure_list.append(structure_list[-1])
        kitt_mnist['structure'].append(structure_list)
        kitt_mnist['net'].append(net['net'])

    kitt_mnist['acc_mean'] = np.mean(kitt_mnist['acc'], axis=0)
    kitt_mnist['acc_std'] = np.std(kitt_mnist['acc'], axis=0)
    kitt_mnist['n_syn_mean'] = np.mean(kitt_mnist['n_syn'], axis=0)
    kitt_mnist['n_syn_std'] = np.std(kitt_mnist['n_syn'], axis=0)
    kitt_mnist['structure_mean'] = np.mean([structure for structure in kitt_mnist['structure']], axis=0)

    for i, structure in enumerate(kitt_mnist['structure_mean']):
        kitt_mnist['structure_mean'][i] = [int(nn) for nn in structure.tolist()]

    print '\n ## Loaded mnist pruned nets ('+str(n_obs)+' obs) ## --------------------'
    print '@ Pruning steps:\t', max_len
    print '@ Mean n synapses:\t', kitt_mnist['n_syn_mean'][-1]
    print '@ Mean structure:\t', kitt_mnist['structure_mean'][-1]
    print '@ Mean accuracy:\t', kitt_mnist['acc_mean'][-1]
    print '----------------------------------------------------'

    return kitt_mnist


if __name__ == '__main__':
    nets = load_mnist()

    net = nets['net'][-1]
    structure = net[0]
    features = range(1, structure[0] + 1)
    classes = range(1, structure[2] + 1)
    syn_exist = net[4]
    class_h = dict()
    for a_class in classes:    # classes: 1 to n
        class_h[a_class] = dict()
        class_h[a_class]['hiddens'] = list()
        class_h[a_class]['inputs'] = list()
        for h_k in range(structure[1]):   # hidden neurons 0 to k
            if bool(syn_exist[1][a_class-1][h_k]):
                class_h[a_class]['hiddens'].append(h_k)
        #print 'hidden neurons for class', a_class, ':', class_h[a_class]['hiddens']

        for h_k in class_h[a_class]['hiddens']:
            class_h[a_class][h_k] = dict()
            class_h[a_class][h_k]['inputs'] = list()
            for i_k in range(structure[0]): # input neurons 0 to k
                if bool(syn_exist[0][h_k][i_k]):
                    class_h[a_class][h_k]['inputs'].append(i_k)
            class_h[a_class]['inputs'] += class_h[a_class][h_k]['inputs']
            #print 'input neurons for class', a_class, 'and hidden', h_k, ':', class_h[a_class][h_k]['inputs']
        #print 'input neurons for class', a_class, ':', len(class_h[a_class]['inputs']), len(np.unique(class_h[a_class]['inputs']))

        # count individual inputs for each output
        class_h[a_class]['n_inputs'] = list()
        for i_k in range(structure[0]):
            class_h[a_class]['n_inputs'].append(class_h[a_class]['inputs'].count(i_k))
        print '\n\nclass', a_class, ': n synapses for individual inputs', class_h[a_class]['n_inputs']

    '''
    # Class vs Feature
    mat = list()
    for a_class in classes:
        mat.append(class_h[a_class]['n_inputs'])
    plt.imshow(mat, vmin=0, vmax=structure[1], aspect='auto')
    #plt.ylim([0, len(features)])
    plt.yticks([c-1 for c in classes], classes)
    plt.xlabel('features')
    plt.ylabel('classes')
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('auto')

    #plt.savefig('../../thesis/img/fs_mat.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()


    # features, number of paths, examples

    # loading examples
    sensors_ranges, terrain_types, all_sensors = load_params('sensors_ranges', 'terrain_types', 'sensors')
    terrains = [terrain_types[str(i)] for i in [6, 8, 15]]
    data = read_data(noises=['no_noise'], terrains=terrains, sensors=all_sensors, n_samples=10)
    samples = dict()
    for terrain in terrains:
        samples[terrain] = [[] for i in range(len(data['no_noise'][terrain][all_sensors[0]]))]
        for sensor in all_sensors:
            for i_sample, sample_terrain in enumerate(data['no_noise'][terrain][sensor]):
                samples[terrain][i_sample] += prepare_signal(signal=sample_terrain[10:40 + 10], sen=sensor)

    n_syn_by_feature = list()
    for f in features:
        n_syn_by_feature.append(sum([class_h[a_class]['n_inputs'][f-1] for a_class in classes]))

    print '\n\n in total: n synapses for individual inputs', n_syn_by_feature
    fig, ax1 = plt.subplots()
    for f_i, n_syn in enumerate(n_syn_by_feature):
        ax1.bar(left=f_i, height=n_syn, width=1, color='darkblue', edgecolor='darkblue')
    ax1.set_xlabel('features')
    ax1.set_ylabel('number of paths to output layer', color='darkblue')
    ax1.set_ylim([0, 280])
    for tl in ax1.get_yticklabels():
        tl.set_color('darkblue')

    colors = ('green', 'brown', 'violet')
    ax2 = ax1.twinx()
    for color, terrain in zip(colors, terrains):
        ax2.plot(choice(samples[terrain]), color=color, label=terrain)
    ax2.set_ylabel('examples of several classes', color='black')
    ax2.set_ylim([-0.25, 1.0])
    for tl in ax2.get_yticklabels():
        tl.set_color('black')

    plt.legend(loc='upper left', ncol=3)
    plt.grid()
    plt.show()

    # features vs classes : at least one connection
    fig, ax1 = plt.subplots()
    for a_class in classes:
        features_used = np.unique(class_h[a_class]['inputs']).tolist()
        features_not_used = [f for f in range(len(features)) if f not in features_used]
        print a_class, ': features used/not used:', len(features_used), '/', len(features_not_used)
        if a_class == 1:
            ax1.plot(features_not_used, [a_class] * len(features_not_used), 'o', color='gray', label='not used features')
            ax1.plot(features_used, [a_class] * len(features_used), 'o', color='red', label='used features')
        else:
            ax1.plot(features_not_used, [a_class] * len(features_not_used), 'o', color='gray')
            ax1.plot(features_used, [a_class] * len(features_used), 'o', color='red')

    ax1.set_ylabel('classes')
    #ax1.set_xlabel('features : Thoraco-Coxa joint sensors')
    #ax1.set_xlabel('features : Coxa-Trochanteral joint sensors')
    #ax1.set_xlabel('features : Femur-Tibia joint sensors')
    ax1.set_xlabel('features : Tactile sensors')
    ax1.set_ylim([-2, len(classes)+3])
    ax1.set_yticks(classes)
    ax1.set_yticklabels([terrain_types[str(i)] for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15)])

    colors = ('green', 'brown', 'violet')
    ax2 = ax1.twinx()
    for color, terrain in zip(colors, terrains):
        ax2.plot(choice(samples[terrain]), color=color)
    ax2.set_ylabel('examples of several classes', color='darkgreen')
    ax2.set_ylim([-0.25, 9.0])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([0, 1])
    for tl in ax2.get_yticklabels():
        tl.set_color('darkgreen')

    base = 720
    plt.xlim([base+1, base+240])
    #plt.xticks([base+20+i*40 for i in range(6)], ('atr_f', 'atr_m', 'atr_h', 'atl_f', 'atl_m', 'atl_h'))
    #plt.xticks([base+20 + i * 40 for i in range(6)], ('acr_f', 'acr_m', 'acr_h', 'acl_f', 'acl_m', 'acl_h'))
    #plt.xticks([base + 20 + i * 40 for i in range(6)], ('afr_f', 'afr_m', 'afr_h', 'afl_f', 'afl_m', 'afl_h'))
    plt.xticks([base + 20 + i * 40 for i in range(6)], ('fr_f', 'fr_m', 'fr_h', 'fl_f', 'fl_m', 'fl_h'))
    plt.grid()
    ax1.legend(loc='upper right', ncol=2)
    plt.show()
    '''


    # Evaluation of feature selection on MNIST
    for a_class in classes:
        digit_interest_image = [255 if set(hiddens_followed_by_digit[digit]).intersection(data[cutting_steps[-1]]['influenced_neurons_by_i0_neuron'][key]) else 0 for key in neurons_ids_input_layer]
        img = Image.new("L", (28, 28), "white")
        img.putdata(digit_interest_image)
        img.save('results_plots/mnist_pixels_interesting_for_digit_'+str(digit)+'.png')