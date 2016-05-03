import matplotlib.pyplot as plt


def plot_all_terrains_by_sensor():
    #terrains = ('rough', 'smooth', 'sliding', 'soft', 'elastic', 'high')
    #terrains = ('mud', 'grass', 'concrete', 'ice', 'gravel', 'sand')
    terrains = ('1', '2', '3')
    sensors = ('atr_f', 'atr_m', 'atr_h', 'atl_f', 'atl_m', 'atl_h', 'acr_f', 'acr_m', 'acr_h', 'acl_f', 'acl_m', 'acl_h',
               'afr_f', 'afr_m', 'afr_h', 'afl_f', 'afl_m', 'afl_h', 'fr_f', 'fr_m', 'fr_h', 'fl_f', 'fl_m', 'fl_h')
    colors = ('r', 'g', 'b', 'c', 'm', 'y', 'k', 'o', 'p')
    data = dict()

    for terrain in terrains:
        data[terrain] = dict()
        with open('../data_examples/data_'+terrain+'.txt', 'r') as data_file:
            data[terrain]['data_str'] = data_file.read()
        for s_i, sensor in enumerate(sensors):
            data[terrain][sensor] = [0.0]
            for line in data[terrain]['data_str'].split('\n')[:-1]:
                values = line.split(';')
                data[terrain][sensor].append(float(values[s_i+1]))

    for s_i, sensor in enumerate(sensors):
        plt.figure('Sensor '+sensor, figsize=(15, 9))
        plt.title('Sensor: '+sensor)
        for t_i, terrain in enumerate(terrains):
            plt.plot(range(len(data[terrain][sensor])), data[terrain][sensor], colors[t_i], label=terrain)
        if s_i >= 18:
            plt.ylim([-0.1, 1.1])
        else:
            plt.ylim([-0.5, 0.5])
        plt.ylabel('sensor_value')
        plt.xlabel('timesteps')
        plt.legend()
        plt.grid()
        #plt.savefig('../sample_examples/terrains_by_sensor/virtual_terrains/'+sensor+'.png')
        plt.show()


def plot_all_sensors_by_terrain():
    with open('../data_examples/data_rough.txt', 'r') as data_file:
        data_str = data_file.read()

    # thoraco joints
    atr_f = [0.0]
    atr_m = [0.0]
    atr_h = [0.0]
    atl_f = [0.0]
    atl_m = [0.0]
    atl_h = [0.0]

    # coxa joints
    acr_f = [0.0]
    acr_m = [0.0]
    acr_h = [0.0]
    acl_f = [0.0]
    acl_m = [0.0]
    acl_h = [0.0]

    # femur joints
    afr_f = [0.0]
    afr_m = [0.0]
    afr_h = [0.0]
    afl_f = [0.0]
    afl_m = [0.0]
    afl_h = [0.0]

    # foot sensors
    fr_f = [0.0]
    fr_m = [0.0]
    fr_h = [0.0]
    fl_f = [0.0]
    fl_m = [0.0]
    fl_h = [0.0]

    for line in data_str.split('\n')[:-1]:
        data = line.split(';')
        # thoraco joints
        atr_f.append(float(data[1]))
        atr_m.append(float(data[2]))
        atr_h.append(float(data[3]))
        atl_f.append(float(data[4]))
        atl_m.append(float(data[5]))
        atl_h.append(float(data[6]))

        # coxa joints
        acr_f.append(float(data[7]))
        acr_m.append(float(data[8]))
        acr_h.append(float(data[9]))
        acl_f.append(float(data[10]))
        acl_m.append(float(data[11]))
        acl_h.append(float(data[12]))

        # femur joints
        afr_f.append(float(data[13]))
        afr_m.append(float(data[14]))
        afr_h.append(float(data[15]))
        afl_f.append(float(data[16]))
        afl_m.append(float(data[17]))
        afl_h.append(float(data[18]))

        # foot sensors
        fr_f.append(float(data[19]))
        fr_m.append(float(data[20]))
        fr_h.append(float(data[21]))
        fl_f.append(float(data[22]))
        fl_m.append(float(data[23]))
        fl_h.append(float(data[24]))

    fig = plt.figure()
    plt.suptitle('(roughness, slip, hardness, elasticity, height) = (1.0, 0.0, 100.0, 0.0, 0.07) # High')
    sp1 = fig.add_subplot(8, 3, 1)
    sp2 = fig.add_subplot(8, 3, 2)
    sp3 = fig.add_subplot(8, 3, 3)
    sp4 = fig.add_subplot(8, 3, 4)
    sp5 = fig.add_subplot(8, 3, 5)
    sp6 = fig.add_subplot(8, 3, 6)
    sp7 = fig.add_subplot(8, 3, 7)
    sp8 = fig.add_subplot(8, 3, 8)
    sp9 = fig.add_subplot(8, 3, 9)
    sp10 = fig.add_subplot(8, 3, 10)
    sp11 = fig.add_subplot(8, 3, 11)
    sp12 = fig.add_subplot(8, 3, 12)
    sp13 = fig.add_subplot(8, 3, 13)
    sp14 = fig.add_subplot(8, 3, 14)
    sp15 = fig.add_subplot(8, 3, 15)
    sp16 = fig.add_subplot(8, 3, 16)
    sp17 = fig.add_subplot(8, 3, 17)
    sp18 = fig.add_subplot(8, 3, 18)
    sp19 = fig.add_subplot(8, 3, 19)
    sp20 = fig.add_subplot(8, 3, 20)
    sp21 = fig.add_subplot(8, 3, 21)
    sp22 = fig.add_subplot(8, 3, 22)
    sp23 = fig.add_subplot(8, 3, 23)
    sp24 = fig.add_subplot(8, 3, 24)
    sp1.set_title('Joint sesor TRf')
    sp1.plot(atr_f)
    sp2.set_title('Joint sesor TRm')
    sp2.plot(atr_m)
    sp3.set_title('Joint sesor TRh')
    sp3.plot(atr_h)
    sp4.set_title('Joint sesor TLf')
    sp4.plot(atl_f)
    sp5.set_title('Joint sesor TLm')
    sp5.plot(atl_m)
    sp6.set_title('Joint sesor TLh')
    sp6.plot(atl_h)
    sp7.set_title('Joint sesor CRf')
    sp7.plot(acr_f)
    sp8.set_title('Joint sesor CRm')
    sp8.plot(acr_m)
    sp9.set_title('Joint sesor CRh')
    sp9.plot(acr_h)
    sp10.set_title('Joint sesor CLf')
    sp10.plot(acl_f)
    sp11.set_title('Joint sesor CLm')
    sp11.plot(acl_m)
    sp12.set_title('Joint sesor CLh')
    sp12.plot(acl_h)
    sp13.set_title('Joint sesor FRf')
    sp13.plot(afr_f)
    sp14.set_title('Joint sesor FRm')
    sp14.plot(afr_m)
    sp15.set_title('Joint sesor FRh')
    sp15.plot(afr_h)
    sp16.set_title('Joint sesor FLf')
    sp16.plot(afl_f)
    sp17.set_title('Joint sesor FLm')
    sp17.plot(afl_m)
    sp18.set_title('Joint sesor FLh')
    sp18.plot(afl_h)
    sp19.set_title('Foot sensor Rf')
    sp19.plot(fr_f)
    sp20.set_title('Foot sensor Rm')
    sp20.plot(fr_m)
    sp21.set_title('Foot sensor Rh')
    sp21.plot(fr_h)
    sp22.set_title('Foot sensor Lf')
    sp22.plot(fl_f)
    sp23.set_title('Foot sensor Lm')
    sp23.plot(fl_m)
    sp24.set_title('Foot sensor Lh')
    sp24.plot(fl_h)
    plt.show()

    '''
    for sensor in sensors:
        plt.figure('Sensor '+sensor, figsize=(15, 9))
        for terrain, color in zip(terrains, colors):
            plt.plot(np.mean(data[terrain][sensor], axis=0), color=color, label=terrain)
        plt.title(sensor+' : mean of 100 samples, no_noise, random init position')
        plt.ylabel('sensor_value')
        plt.xlabel('timesteps')
        plt.legend()
        plt.grid()
        #plt.savefig('../../sample_examples/terrains_by_sensor/virtual_terrains/'+sensor+'_mean_of_100.png')
        #plt.savefig('../../sample_examples/terrains_by_sensor/virtual_terrains/'+sensor+'_mean_of_100.eps')
        plt.show()
    '''

    '''
    plt.figure('NN input : mean of 100 samples', figsize=(15, 9))
    for terrain, color in zip(terrains, colors):
        plt.plot(np.mean(samples[terrain], axis=0), color, label=terrain)
    plt.title('NN input : mean of 100 samples, timesteps [10:90] for every sensor, 24 sensors')
    plt.suptitle('no terrain (samples) noise, signal gaussian noise std: 1e-10 (no_noise)')
    plt.ylabel('sensor_value normed to [0,1]')
    plt.xlabel('timesteps sensor by sensor (NN input neurons)')
    plt.legend()
    plt.grid()
    plt.legend()
    #plt.savefig('../../sample_examples/nn_input_mean_of_100_nonoise.png')
    #plt.savefig('../../sample_examples/nn_input_mean_of_100_nonoise.eps')
    plt.show()
    '''


if __name__ == '__main__':
    #plot_all_sensors_by_terrain()
    plot_all_terrains_by_sensor()
