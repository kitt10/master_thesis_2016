#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from functions import load_net


def load_nets():
    kitt_mnist = {'accs': list(), 'errs': list(), 'times': list(), 'acc_mean': None, 'err_mean': None, 'acc_std': None, 'err_std': None}
    for obs in range(1, 11):
        net = load_net('../cache/trained/kitt_mnist&0.1_100_[15]_'+str(obs)+'.net')
        kitt_mnist['accs'].append(net['training_eval'][0]['v'])
        kitt_mnist['errs'].append(net['training_eval'][1]['v'])
        kitt_mnist['times'].append(net['training_eval'][2])

    kitt_mnist['acc_mean'] = np.mean(kitt_mnist['accs'], axis=0)
    kitt_mnist['acc_std'] = np.std(kitt_mnist['accs'], axis=0)
    kitt_mnist['err_mean'] = np.mean(kitt_mnist['errs'], axis=0)
    kitt_mnist['err_std'] = np.std(kitt_mnist['errs'], axis=0)
    kitt_mnist['time_mean'] = np.mean(kitt_mnist['times'], axis=0)

    print 'kitt on MNIST:'
    print net['skills'][1]

    kitt_xor = {'accs': list(), 'errs': list(), 'times': list(), 'acc_mean': None, 'err_mean': None, 'acc_std': None, 'err_std': None}
    for obs in range(1, 11):
        net = load_net('../cache/trained/kitt_xor&0.5_100_[2]_' + str(obs) + '.net')
        kitt_xor['accs'].append(net['training_eval'][0]['v'])
        kitt_xor['errs'].append(net['training_eval'][1]['v'])
        kitt_xor['times'].append(net['training_eval'][2])

    kitt_xor['acc_mean'] = np.mean(kitt_xor['accs'], axis=0)
    kitt_xor['acc_std'] = np.std(kitt_xor['accs'], axis=0)
    kitt_xor['err_mean'] = np.mean(kitt_xor['errs'], axis=0)
    kitt_xor['err_std'] = np.std(kitt_xor['errs'], axis=0)
    kitt_xor['time_mean'] = np.mean(kitt_xor['times'], axis=0)

    sknn_mnist = {'accs': list(), 'errs': list(), 'times': list(), 'acc_mean': None, 'err_mean': None, 'acc_std': None, 'err_std': None}
    for obs in range(1, 11):
        net = load_net('../cache/trained/sknn_mnist&0.003_100_[10]_' + str(obs) + '.net')
        sknn_mnist['accs'].append(net['training_eval'][0]['v'])
        sknn_mnist['errs'].append(net['training_eval'][1]['v'])
        sknn_mnist['times'].append(net['training_eval'][2])

    sknn_mnist['acc_mean'] = np.mean(sknn_mnist['accs'], axis=0)
    sknn_mnist['acc_std'] = np.std(sknn_mnist['accs'], axis=0)
    sknn_mnist['err_mean'] = np.mean(sknn_mnist['errs'], axis=0)
    sknn_mnist['err_std'] = np.std(sknn_mnist['errs'], axis=0)
    sknn_mnist['time_mean'] = np.mean(sknn_mnist['times'], axis=0)

    print 'sknn on MNIST:'
    print net['skills'][1]

    sknn_xor = {'accs': list(), 'errs': list(), 'times': list(), 'acc_mean': None, 'err_mean': None, 'acc_std': None, 'err_std': None}
    for obs in range(1, 11):
        net = load_net('../cache/trained/sknn_xor&0.3_100_[2]_' + str(obs) + '.net')
        sknn_xor['accs'].append(net['training_eval'][0]['v'])
        sknn_xor['errs'].append(net['training_eval'][1]['v'])
        sknn_xor['times'].append(net['training_eval'][2])

    #print [len(xi) for xi in sknn_xor['accs']]

    sknn_xor['acc_mean'] = np.mean(sknn_xor['accs'], axis=0)
    sknn_xor['acc_std'] = np.std(sknn_xor['accs'], axis=0)
    sknn_xor['err_mean'] = np.mean(sknn_xor['errs'], axis=0)
    sknn_xor['err_std'] = np.std(sknn_xor['errs'], axis=0)
    sknn_xor['time_mean'] = np.mean(sknn_xor['times'], axis=0)

    return kitt_mnist, sknn_mnist, kitt_xor, sknn_xor



if __name__ == '__main__':
    #kitt_net = load_net('../cache/trained/kitt_mnist&0.1_10_[15].net')
    #sknn_net = load_net('../cache/trained/sknn_mnist&0.1_10_[15].net')
    sknn_mnist_cpu = load_net('../cache/trained/sknn_mnist&0.3_5_[10]_cpu.net')
    sknn_xor_cpu = load_net('../cache/trained/sknn_xor&0.3_100_[2]_cpu.net')

    kitt_mnist, sknn_mnist, kitt_xor, sknn_xor = load_nets()

    ''' Accuracy through epochs '''
    plt.figure()
    plt.errorbar(x=range(101), y=kitt_mnist['acc_mean'], yerr=kitt_mnist['acc_std'], label='kitt on MNIST',
                 color='blue')
    plt.errorbar(x=range(101), y=kitt_xor['acc_mean'], yerr=kitt_xor['acc_std'], label='kitt on XOR',
                 color='cyan')
    plt.errorbar(x=range(101), y=sknn_mnist['acc_mean'], yerr=sknn_mnist['acc_std'], label='sknn on MNIST',
                 color='green')
    plt.errorbar(x=range(101), y=sknn_xor['acc_mean'], yerr=sknn_xor['acc_std'], label='sknn on XOR',
                 color='lime')
    plt.xlabel('training epoch')
    plt.xlim([-1, 101])
    plt.ylabel('classification accuracy')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(loc='best', ncol=2)
    plt.savefig('../../thesis/img/kitt_verify_acc.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ''' Error through epochs '''
    plt.figure()
    '''plt.errorbar(x=range(1, 101), y=kitt_mnist['err_mean'], yerr=kitt_mnist['err_std'], label='kitt on MNIST',
                 color='red')'''
    plt.errorbar(x=range(1, 101), y=kitt_xor['err_mean'], yerr=kitt_xor['err_std'], label='kitt on XOR',
                 color='violet')
    '''plt.errorbar(x=range(1, 101), y=sknn_mnist['err_mean'], yerr=sknn_mnist['err_std'], label='sknn on MNIST',
             color='brown')'''
    plt.errorbar(x=range(1, 101), y=sknn_xor['err_mean'], yerr=sknn_xor['err_std'], label='sknn on XOR',
                 color='gold')
    plt.xlabel('training epoch')
    plt.xlim([-1, 101])
    plt.ylabel('classification error')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(loc='best', ncol=2)
    plt.savefig('../../thesis/img/kitt_verify_err.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ''' Average epoch time '''
    plt.figure()
    plt.bar(1, np.mean(kitt_mnist['time_mean']), width=1, color='blue')
    plt.bar(2, np.mean(sknn_mnist_cpu['training_eval'][2]), width=1, color='green')
    plt.bar(3, np.mean(sknn_mnist['time_mean']), width=1, color='yellow')
    plt.xlim([0, 5])
    plt.xticks([1.5, 2.5, 3.5], ['kitt', 'sknn:cpu', 'sknn:gpu'], ha='center')
    plt.ylabel('average epoch time [s]')
    plt.grid()
    plt.savefig('../../thesis/img/kitt_verify_time.eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()

