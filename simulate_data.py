'''
simulate the data
'''

import os
import shutil
from subprocess import Popen, PIPE
from utility import Utility

if __name__ == '__main__':

    vary_power = 5

    # 1. generate 5 train and 5 test   [output_filename, sample_label, sensor_density, random_seed, vary_power]
    config = [['100train', 2, 100, 0,  vary_power],  ['100test', 1, 100, 100, vary_power],
              ['101train', 2, 300, 0,  vary_power],  ['101test', 1, 300, 100, vary_power],
              ['102train', 2, 500, 0,  vary_power],  ['102test', 1, 500, 100, vary_power],
              ['103train', 2, 700, 0,  vary_power],  ['103test', 1, 700, 100, vary_power],
              ['104train', 2, 1000, 0, vary_power], ['104test', 1, 1000, 100, vary_power],
            ]
    template = 'python generate.py -gd -rd data/{} -sl {} -sd {} -rs {} -nt 10 -ntup -mind 1 -vp {}'
    ps = []
    for c in config:
        command = template.format(c[0], c[1], c[2], c[3], c[4])
        print(command)
        p = Popen(command, shell=True, stdout=PIPE)
        ps.append(p)
    for p in ps:
        p.wait()

    # 2. generate the IPSN format data
    template = 'python generate.py -ipsn -rd data/{} -sd {} -rs {}'
    ps = []
    for c in config[1::2]:
        command = template.format(c[0], c[2], c[3])
        p = Popen(command, shell=True, stdout=PIPE)
        ps.append(p)
    for p in ps:
        p.wait()

    # 3. merge the data of step 1
    train_dir = 'data/105train'
    Utility.remove_make(train_dir)
    size = 9216
    for i in range(size):
        if i % 100 == 0:
            print(i)
        folder = format(i, '06d')
        dest_dir = os.path.join(train_dir, folder)
        Utility.remove_make(dest_dir)
        counter = 0
        for c in config[::2]:
            for j in range(c[1]):
                source = os.path.join('data', c[0], folder, str(j))
                shutil.copy(source + '.npy',        os.path.join(dest_dir, str(counter) + '.npy'))
                shutil.copy(source + '.target.npy', os.path.join(dest_dir, str(counter) + '.target.npy'))
                counter += 1

    test_dir = 'data/105test'
    Utility.remove_make(test_dir)
    for i in range(size):
        folder = format(i, '06d')
        dest_dir = os.path.join(test_dir, folder)
        Utility.remove_make(dest_dir)
        counter = 0
        for c in config[1::2]:
            for j in range(c[1]):
                source = os.path.join('data', c[0], folder, str(j))
                shutil.copy(source + '.npy',        os.path.join(dest_dir, str(counter) + '.npy'))
                shutil.copy(source + '.target.npy', os.path.join(dest_dir, str(counter) + '.target.npy'))
                shutil.copy(source + '.json',       os.path.join(dest_dir, str(counter) + '.json'))
                counter += 1

