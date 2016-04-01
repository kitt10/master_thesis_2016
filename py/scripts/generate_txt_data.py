#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.generate_txt_data
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    This script runs the Amos II simulation on specific terrains and saves sensory data as .txt files.

    @arg n_jobs         : Number of simulation runs for each terrain.
    @arg terrains       : Terrains to be simulated.
    @arg noise          : Terrain noise.
    @arg n_timesteps    : Number of simulation steps over one experiment.
    @arg gait           : Amos II gait.
"""

import argparse
import subprocess
import os
from signal import SIGTERM
from time import sleep
from shutil import copyfile
from functions import load_params


def parse_arguments():
    parser = argparse.ArgumentParser(description='Runs the Amos II simulation a saves sensor data as .txt.')
    parser.add_argument('-nj', '--n_jobs', type=int, default=100,
                        help='Number of simulation runs')
    parser.add_argument('-t', '--terrains', type=int, default=range(1, 16), nargs='+', choices=range(1, 16),
                        help='Terrains to be generated (integers)')
    parser.add_argument('-n', '--noise', type=str, default='no_noise', choices=noise_params.keys(),
                        help='Terrain noise type')
    parser.add_argument('-nt', '--n_timesteps', type=int, default=100,
                        help='Number of simulation steps')
    parser.add_argument('-g', '--gait', type=str, default='tripod', choices=['tripod'],
                        help='Amos II gait')
    parser.add_argument('-sn', '--sim_noise', type=float, default=0.0,
                        help='Amos II simulation noise')
    return parser.parse_args()


if __name__ == '__main__':
    terrain_types, noise_params = load_params('terrain_types', 'noise_types')
    args = parse_arguments()

    noise_type, (noise_prefix, noise_param) = args.noise, noise_params[args.noise]
    gait = args.gait
    n_jobs = args.n_jobs
    n_timesteps = args.n_timesteps
    sim_noise = args.sim_noise
    terrains_i = sorted(args.terrains)
    terrains_to_use = [terrain_types[str(i)] for i in terrains_i]

    os.chdir('../../simulation/mbulinai22015-gorobots_edu-fork/practices/amosii')

    for i_terrain, terrain_type in zip(terrains_i, terrains_to_use):
        destination_dir = '../../../../data/'+noise_type+'/'+noise_prefix+terrain_type+'/'
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        n_generated = len([name for name in os.listdir(destination_dir) if os.path.isfile(destination_dir+name)])

        for i_job in range(1, n_jobs+1):
            command = './start '+str(i_terrain)+' '+str(noise_param)+' '+str(sim_noise)
            print 'job:', i_job, '/', n_jobs, '(generated: '+str(n_generated+i_job-1)+')', '##', terrain_type, ',', \
                gait, ',', noise_type, ',',  n_timesteps, command
            with open(os.devnull, 'w') as shut_up:
                sp = subprocess.Popen(command,
                                      stdout=shut_up, stderr=shut_up, shell=True, preexec_fn=os.setsid)
            sleep(n_timesteps/10)
            os.killpg(os.getpgid(sp.pid), SIGTERM)

            copyfile('data.txt', destination_dir+
                     noise_prefix+terrain_type+'_'+gait+'_job'+str(n_generated+i_job).zfill(4)+'.txt')
