'''
Assumption that regression 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from typing import List
from IPython.display import clear_output

import sys
sys.path.insert(0, '/home/caitao/Project/dl-localization')
from input_output import Default
from utility import Utility
from deepleaning_models import Net3


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
            matrix = self.lower + matrix * (self.upper - self.lower)  # might have zero in the denominator
        return matrix.astype(np.float32)


class Metrics:
    '''Evaluation metrics
    '''
    @staticmethod
    def localization_error_regression(pred_batch, truth_batch, debug=False):
        '''euclidian error when modeling the output representation is a matrix (image)\
           both pred and truth are batches, typically a batch of 32
        '''
        error = []
        for pred, truth in zip(pred_batch, truth_batch):
            if debug:
                print(pred, truth)
            pred_x, pred_y = pred[0], pred[1]
            true_x, true_y = truth[0], truth[1]
            error.append(Utility.distance((pred_x, pred_y), (true_x, true_y)))
        return error

# data

class SensorInputDatasetRegression(Dataset):
    '''Sensor reading input dataset
       Output is image, model as a image segmentation problem
    '''
    def __init__(self, root_dir: str, grid_len: int, transform=None):
        '''
        Args:
            root_dir:  directory with all the images
            labels:    labels of images
            transform: optional transform to be applied on a sample
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.length = len(os.listdir(self.root_dir))
        self.sample_per_label = self.get_sample_per_label()
        self.grid_len = grid_len

    def __len__(self):
        return self.length * self.sample_per_label

    def __getitem__(self, idx):
        folder = int(idx/self.sample_per_label)
        folder = format(folder, '06d')
        matrix_name = str(idx%self.sample_per_label) + '.npy'
        target_name = str(idx%self.sample_per_label) + '.target.npy'
        matrix_path = os.path.join(self.root_dir, folder, matrix_name)
        matrix = np.load(matrix_path)
        if self.transform:
            matrix = self.transform(matrix)
        target_arr = self.get_regression_target(folder, target_name)
        target_arr = self.min_max_normalize(target_arr)
        sample = {'matrix':matrix, 'label':target_arr}
        return sample

    def get_sample_per_label(self):
        folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
        samples = glob.glob(os.path.join(folder, '*.npy'))
        targets = glob.glob(os.path.join(folder, '*.target.npy'))
        return len(samples) - len(targets)

    def get_regression_target(self, folder: str, target_name: str):
        '''
        Args:
            folder: example of folder is 000001
        Return:
            a two dimension matrix
        '''
        target_file = os.path.join(self.root_dir, folder, target_name)
        target = np.load(target_file)
        return target.astype(np.float32)

    def min_max_normalize(self, target_arr: np.ndarray):
        '''scale the localization to a range of (0, 1)
        '''
        target_arr /= self.grid_len
        return target_arr

    def undo_normalize(self, arr: np.ndarray):
        arr *= self.grid_len
        return arr


tf = T.Compose([
     MinMaxNormalize(),
     T.ToTensor()
])


def train_test(train, test, num_epoch, net):
    '''
    Args:
        the filenames of training and testing dataset
    Return:
        best testing error of the all epochs
    '''
    # training
    train = os.path.join('.', 'data', train)
    sensor_input_dataset = SensorInputDatasetRegression(root_dir = train, grid_len = Default.grid_length, transform = tf)
    sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=32, shuffle=True, num_workers=3)
    # testing
    test = os.path.join('.', 'data', test)
    sensor_input_test_dataset = SensorInputDatasetRegression(root_dir = test, grid_len = Default.grid_length, transform = tf)
    sensor_input_test_dataloader = DataLoader(sensor_input_test_dataset, batch_size=32, shuffle=True, num_workers=3)

    print(net)

    device     = torch.device('cuda')
    model      = net.to(device)
    optimizer  = optim.Adam(model.parameters(), lr=0.001)
    criterion  = nn.MSELoss()  # criterion is the loss function

    num_epochs = num_epoch
    train_losses_epoch = []
    train_errors_epoch = []
    test_errors_epoch  = []
    print_every = 500

    for epoch in range(num_epochs):
        print(f'epoch = {epoch}')
        train_losses = []
        train_errors = []
        test_errors  = []
        model.train()
        for t, sample in enumerate(sensor_input_dataloader):
            X = sample['matrix'].to(device)
            y = sample['label'].to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pred = pred.data.cpu()
            y = y.data.cpu()
            train_errors.extend(Metrics.localization_error_regression(pred, y))
            if t % print_every == 0:
                print(f't = {t}, loss = {loss.item()}')

        model.eval()
        for t, sample in enumerate(sensor_input_test_dataloader):
            X = sample['matrix'].to(device)
            y = sample['label'].to(device)
            pred = model(X)
            pred = pred.data.cpu()
            y    = y.data.cpu()
            pred = sensor_input_dataset.undo_normalize(pred)
            y    = sensor_input_dataset.undo_normalize(y)
            test_errors.extend(Metrics.localization_error_regression(pred, y))

        print('train loss mean =', np.mean(train_losses))
        print('train loss std  =', np.std(train_losses))
        print('train mean =', np.mean(train_errors))
        print('train std  =', np.std(train_errors))    
        print('test mean =', np.mean(test_errors))
        print('test std  =', np.std(test_errors))
        train_losses_epoch.append((np.mean(train_losses), np.std(train_losses)))
        train_errors_epoch.append((np.mean(train_errors), np.std(train_errors)))
        test_errors_epoch.append((np.mean(test_errors), np.std(test_errors)))

    print('train loss')
    for loss in train_losses_epoch:
        print(loss)
    print('train error')
    for error in train_errors_epoch:
        print(error)
    print('test error')
    for error in test_errors_epoch:
        print(error)
    return sorted(test_errors_epoch, key=lambda x:x[0])[0]


if __name__ == '__main__':
    # time.sleep(60*60*2)
    regression = []
    training_dataset = ['matrix-train20', 'matrix-train21', 'matrix-train22', 'matrix-train23', 'matrix-train24']
    testing_dataset  = ['matrix-test20',  'matrix-test20',  'matrix-test20',  'matrix-test20',  'matrix-test20']
    epoches = [10, 20, 30, 40, 50]
    for train, test, epoch in zip(training_dataset, testing_dataset, epoches):
        tmp = []
        for i in range(3):
            print(train, i)
            net = Net3()
            tmp.append(np.array(train_test(train, test, epoch, net)))
        regression.append(np.mean(tmp, axis=0))
        print(regression)
    np.savetxt('experimental/regression5.txt', np.array(regression), delimiter=',')
