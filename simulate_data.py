'''
simulate the data for deepmtl and deeptxfinder
'''

import os
import shutil
import glob
import numpy as np
from subprocess import Popen, PIPE
from utility import Utility


# if __name__ == '__main__':
if __name__ == '':

    vary_power = 2

    # 1. generate 5 train and 5 test   [output_filename, sample_label, sensor_density, random_seed, vary_power]
    print('step 1 ...')
    config = [['200train', 2, 200, 0,  vary_power],  ['200test', 1, 200, 100, vary_power],
              ['201train', 2, 300, 0,  vary_power],  ['201test', 1, 400, 100, vary_power],
              ['202train', 2, 600, 0,  vary_power],  ['202test', 1, 600, 100, vary_power],
              ['203train', 2, 800, 0,  vary_power],  ['203test', 1, 800, 100, vary_power],
              ['204train', 2, 1000, 0, vary_power],  ['204test', 1, 1000, 100, vary_power],
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
    print('step 2 ...')
    template = 'python generate.py -ipsn -rd data/{} -sd {} -rs {}'
    ps = []
    for c in config[1::2]:  # test
        command = template.format(c[0], c[2], c[3])
        p = Popen(command, shell=True, stdout=PIPE)
        ps.append(p)
    for p in ps:
        p.wait()

    # 3. merge the data of step 1
    print('step 3 ...')
    train_dir = 'data/205train'
    Utility.remove_make(train_dir)
    size = 9216
    for i in range(size):
        if i % 1000 == 0:
            print(i)
        folder = format(i, '06d')
        dest_dir = os.path.join(train_dir, folder)
        Utility.remove_make(dest_dir)
        counter = 0
        for c in config[::2]:   # train
            for j in range(c[1]):
                source = os.path.join('data', c[0], folder, str(j))
                shutil.copy(source + '.npy',        os.path.join(dest_dir, str(counter) + '.npy'))
                shutil.copy(source + '.target.npy', os.path.join(dest_dir, str(counter) + '.target.npy'))
                counter += 1

    test_dir = 'data/205test'
    Utility.remove_make(test_dir)
    for i in range(size):
        if i % 1000 == 0:
            print(i)
        folder = format(i, '06d')
        dest_dir = os.path.join(test_dir, folder)
        Utility.remove_make(dest_dir)
        counter = 0
        for c in config[1::2]:  # test
            for j in range(c[1]):
                source = os.path.join('data', c[0], folder, str(j))
                shutil.copy(source + '.npy',        os.path.join(dest_dir, str(counter) + '.npy'))
                shutil.copy(source + '.target.npy', os.path.join(dest_dir, str(counter) + '.target.npy'))
                shutil.copy(source + '.json',       os.path.join(dest_dir, str(counter) + '.json'))
                counter += 1


if __name__ == '__main__':
    '''for deeptxfinder data, split the dataset into 10 dataset
    '''

    train_dir = 'data/205train/*'
    dest_dirs  = ['data2/cnn2_train1', 'data2/cnn2_train2', 'data2/cnn2_train3', 'data2/cnn2_train4', 'data2/cnn2_train5',\
                 'data2/cnn2_train6', 'data2/cnn2_train7', 'data2/cnn2_train8', 'data2/cnn2_train9', 'data2/cnn2_train10']

    for i, dest_dir in zip(range(1, 11), dest_dirs):
        counter = 0
        for folder in sorted(glob.glob(train_dir)):
            for sample in sorted(glob.glob(folder + '/*.target.npy')):
                target = np.load(sample)
                num_tx = len(target)
                if num_tx != i:
                    continue
                label_folder = format(counter, '06d')
                directory = os.path.join(dest_dir, label_folder)
                Utility.remove_make(directory)
                shutil.copy(sample, os.path.join(directory, '0.target.npy'))  # basically 1 sample per label
                sample = sample.replace('.target', '')
                shutil.copy(sample, os.path.join(directory, '0.npy'))
                counter += 1
        print(f'train, num tx = {i}, counter = {counter}')


    test_dir  = 'data/205test/*'
    dest_dirs  = ['data2/cnn2_test1', 'data2/cnn2_test2', 'data2/cnn2_test3', 'data2/cnn2_test4', 'data2/cnn2_test5',\
                  'data2/cnn2_test6', 'data2/cnn2_test7', 'data2/cnn2_test8', 'data2/cnn2_test9', 'data2/cnn2_test10']
    for folder in dest_dirs:
        Utility.remove_make(folder)

    for i, dest_dir in zip(range(1, 11), dest_dirs):
        counter = 0
        for folder in sorted(glob.glob(test_dir)):
            for sample in sorted(glob.glob(folder + '/*.target.npy')):
                target = np.load(sample)
                num_tx = len(target)
                if num_tx != i:
                    continue
                label_folder = format(counter, '06d')
                directory = os.path.join(dest_dir, label_folder)
                Utility.remove_make(directory)
                shutil.copy(sample, os.path.join(directory, '0.target.npy'))  # basically 1 sample per label
                sample = sample.replace('.target', '')
                shutil.copy(sample, os.path.join(directory, '0.npy'))
                counter += 1
        print(f'test, num tx = {i}, counter = {counter}')
