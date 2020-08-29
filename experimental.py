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
            error.append(Utility.distance_error((pred_x, pred_y), (true_x, true_y)))
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
        matrix_path = os.path.join(self.root_dir, folder, matrix_name)
        matrix = np.load(matrix_path)
        if self.transform:
            matrix = self.transform(matrix)
        label_arr = self.get_regression_output(folder)
        label_arr = self.min_max_normalize(label_arr)
        sample = {'matrix':matrix, 'label':label_arr}
        return sample

    def get_sample_per_label(self):
        folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
        samples = glob.glob(os.path.join(folder, '*.npy'))
        return len(samples)
    
    def get_regression_output(self, folder: str):
        '''
        Args:
            folder: example of folder is 000001
        Return:
            a two dimension matrix
        '''
        twoDstr = glob.glob(os.path.join(self.root_dir, folder, '*.png'))[0]
        twoDstr = twoDstr.split('/')[4]
        twoDstr = twoDstr[1:-5]
        x, y = twoDstr.split(',')
        x, y = int(x), int(y)
        arr = np.array([x, y])
        return arr.astype(np.float32)
    
    def min_max_normalize(self, label_arr: np.ndarray):
        '''scale the localization to a range of (0, 1)
        '''
        label_arr /= self.grid_len
        return label_arr
    
    def undo_normalize(self, arr: np.ndarray):
        arr *= self.grid_len
        return arr


tf = T.Compose([
     MinMaxNormalize(),
     T.ToTensor()
])


class Net3(nn.Module):
    '''the output of the fully connected layer is 1x2 array
       assume the input matrix is 1 x 100 x 100
    '''
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.groupnorm1 = nn.GroupNorm(1, 8)
        self.groupnorm2 = nn.GroupNorm(1, 16)
        self.groupnorm3 = nn.GroupNorm(1, 32)
        self.fc1   = nn.Linear(2592, 100)
        self.fc2   = nn.Linear(100, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.groupnorm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.groupnorm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.groupnorm3(self.conv3(x))), 2)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    return sorted(test_errors_epoch, key=lambda x:x[0])[0][0]


if __name__ == '__main__':
    time.sleep(60*60*2)
    regression = []
    training_dataset = ['matrix-train10', 'matrix-train11', 'matrix-train12', 'matrix-train13', 'matrix-train14']
    testing_dataset  = ['matrix-test10',  'matrix-test10',  'matrix-test10',  'matrix-test10',  'matrix-test10']
    epoches = [10, 20, 30, 30, 30]
    for train, test, epoch in zip(training_dataset, testing_dataset, epoches):
        tmp = []
        for i in range(3):
            print(train, i)
            net = Net3()
            tmp.append(train_test(train, test, epoch, net))
        regression.append(np.mean(tmp))
        print(regression)
    np.savetxt('experimental/regression3.txt', np.array(regression), delimiter=',')
