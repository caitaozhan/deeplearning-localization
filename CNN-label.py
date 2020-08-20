import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from typing import List
from IPython.display import clear_output

import sys
sys.path.insert(0, '/home/caitao/Project/dl-localization')
from input_output import Default


# when the input is a matrix
class SensorInputDataset(Dataset):
    '''Sensor reading input dataset'''
    def __init__(self, root_dir: str, transform=None):
        '''
        Args:
            root_dir:  directory with all the images
            labels:    labels of images
            transform: optional transform to be applied on a sample
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))

    def __len__(self):
        return self.length * Default.sample_per_label

    def __getitem__(self, idx):
        folder   = self.oneDint2twoDstr(int(idx/Default.sample_per_label))
        matrix_name = str(idx%Default.sample_per_label) + '.txt'
        matrix_path = os.path.join(self.root_dir, folder, matrix_name)
        matrix = np.loadtxt(matrix_path, delimiter=',', dtype=float)
        if self.transform:
            matrix = self.transform(matrix)
        label = self.twoDstr2oneDint(folder)
        sample = {'matrix':matrix, 'label':label}
        return sample

    def oneDint2twoDstr(self, oneDint):
        '''convert a one dimension integer index to a two dimension string index'''
        x = oneDint // Default.grid_length
        y = oneDint % Default.grid_length
        return f'({x}, {y})'

    def twoDstr2oneDint(self, twoDstr):
        '''convert a two dimension string to a one dimension integet index for the labels'''
        twoDstr = twoDstr[1:-1]
        x, y = twoDstr.split(',')
        x, y = int(x), int(y)
        return x*Default.grid_length + y


class MinMaxNormalize:
    '''Min max normalization, the new bound is (lower, upper)
    '''
    def __init__(self, lower=0, upper=1):
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
        self.lower = lower
        self.upper = upper

    def __call__(self, matrix):
        minn = matrix.min()
        maxx = matrix.max()
        matrix = (matrix - minn) / (maxx - minn)
        if self.lower != 0 or self.upper != 1:
            matrix = self.lower + matrix * (self.upper - self.lower)
        return matrix.astype(np.float32)


def main():
    '''main'''
    root_dir = './data/matrix-1'
    tf = T.Compose([
        MinMaxNormalize(),
        T.ToTensor()
    ])

    sensor_input_dataset = SensorInputDataset(root_dir = root_dir, transform = tf)
    # sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=4, shuffle=True, num_workers=4)
    print(sensor_input_dataset[0]['matrix'][0][0])


main()
