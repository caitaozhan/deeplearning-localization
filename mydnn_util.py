'''utilities that the CNN models needs
'''

import torchvision.transforms as T
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from input_output import Default
from utility import Utility


class SensorInputDatasetTranslation(Dataset):
    '''Sensor reading input dataset -- for multi TX
       Output is image, model as a image segmentation problem
    '''
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
        self.sample_per_label = self.get_sample_per_label()

    def __len__(self):
        return self.length * self.sample_per_label

    def __getitem__(self, idx):
        folder = int(idx/self.sample_per_label)
        folder = format(folder, '06d')
        matrix_name = str(idx%self.sample_per_label) + '.npy'
        matrix_path = os.path.join(self.root_dir, folder, matrix_name)
        target_name = str(idx%self.sample_per_label) + '.target.npy'
        target_img, target_float = self.get_translation_target(folder, target_name)
        matrix = np.load(matrix_path)
        if self.transform:
            matrix = self.transform(matrix)
        target_num = len(target_float)
        power_name = str(idx%self.sample_per_label) + '.power.npy'
        power_path = os.path.join(self.root_dir, folder, power_name)
        power = np.load(power_path)
        sample = {'matrix':matrix, 'target':target_img, 'target_float':target_float, 'target_num':target_num, 'power': power, 'index':idx}
        return sample

    def get_sample_per_label(self):
        folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
        samples = glob.glob(os.path.join(folder, '*.npy'))
        targets = glob.glob(os.path.join(folder, '*.target.npy'))
        powers = glob.glob(os.path.join(folder, '*.power.npy'))
        return len(samples) - len(targets) - len(powers)

    def get_translation_target(self, folder: str, target_name: str):
        '''
        Args:
            folder      -- eg. 000001
            target_name -- eg. 0.target.npy
        Return:
            np.ndarray, n = 2, the pixels surrounding TX will be assigned some values
        '''
        location = np.load(os.path.join(self.root_dir, folder, target_name))
        num_tx = len(location)
        grid = np.zeros((Default.grid_length, Default.grid_length))
        for i in range(num_tx):
            x, y = location[i][0], location[i][1]
            target_float = (x, y)
            x, y = int(x), int(y)
            neighbor = []
            sum_weight = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    nxt = (x + i, y + j)
                    if 0 <= nxt[0] < Default.grid_length and 0 <= nxt[1] < Default.grid_length:
                        weight = 1./Utility.distance((nxt[0] + 0.5, nxt[1] + 0.5), target_float)
                        sum_weight += weight
                        neighbor.append((nxt, weight))
            for n, w in neighbor:
                grid[n[0]][n[1]] += w / sum_weight * len(neighbor) * 3  # 2 is for adding weights
        grid = np.expand_dims(grid, 0)
        return grid.astype(np.float32), location.astype(np.float32)


class UniformNormalize:
    '''Set a uniform threshold accross all samples
    '''
    def __init__(self, noise_floor):
        self.noise_floor = noise_floor

    def __call__(self, matrix):
        matrix -= self.noise_floor
        matrix /= (-self.noise_floor/2)
        return matrix.astype(np.float32)

class Metrics:
    '''Evaluation metrics
    '''
    @staticmethod
    def localization_error_image_continuous_simple(pred_batch, truth_batch, index, grid_len, peak_threshold, size, debug=False):
        '''Continuous -- for single TX
           euclidian error when modeling the output representation is a matrix (image)
           both pred and truth are batches, typically a batch of 32
           now both prediction and truth are continuous numbers
        Args:
            pred_batch:  numpy.ndarray -- size=(N, 1, 100, 100)
            truth_batch: numpy.ndarray -- size=(N, num_tx, 2)
        Return:
            pred_locs -- list<np.ndarray>
            errors    -- list<list>
            misses    -- list
            false     -- list
        '''
        def float_target(pred_matrix, pred_peaks):
            new_pred_peaks = []
            for pred_x, pred_y in pred_peaks:
                sum_weight = 0
                neighbor = []
                for d in [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]:
                    nxt = (pred_x + d[0], pred_y + d[1])
                    if 0 <= nxt[0] < grid_len and 0 <= nxt[1] < grid_len:
                        neighbor.append(((nxt[0] + 0.5, nxt[1] + 0.5), pred[nxt[0]][nxt[1]]))
                        sum_weight += pred_matrix[nxt[0]][nxt[1]]
                pred_x, pred_y = 0, 0
                for n in neighbor:
                    loc = n[0]
                    w   = n[1]
                    pred_x += loc[0] / sum_weight * w
                    pred_y += loc[1] / sum_weight * w
                new_pred_peaks.append((pred_x, pred_y))
            return new_pred_peaks

        pred_locs, errors, misses, falses = [], [], [], []
        for i, pred, truth, indx in zip(range(len(pred_batch)), pred_batch, truth_batch, index):
            # 1: get the multiple predicted locations
            pred = pred[0]           # there is only one channel
            # pred_peaks, _ = Utility.detect_peak(pred, np.round(pred_n, 0), peak_threshold)         # get the predictions  TIME: 23 milliseconds
            pred_peaks = Utility.detect_peak_simple(pred, peak_threshold=peak_threshold, size=size)         # get the predictions  TIME: 23 milliseconds
            pred_peaks = float_target(pred, pred_peaks)
            pred_locs.append(pred_peaks)
            # 2: do a matching and get the error
            radius_threshold = Default.grid_length * Default.error_threshold
            error, miss, false = Utility.compute_error(pred_peaks, truth, radius_threshold, False)
            errors.append(error)
            misses.append(miss)
            falses.append(false)
            if debug:
                print(i, indx, 'pred', [(round(loc[0], 2), round(loc[1], 2)) for loc in pred_peaks], '; truth', \
                      [(round(loc[0], 2), round(loc[1], 2)) for loc in truth], ' ; error', error, ' ; miss', miss, ' ; false', false)
        return pred_locs, errors, misses, falses



    @staticmethod
    def localization_error_image_continuous(pred_batch, pred_ntx, truth_batch, index, grid_len, peak_threshold, debug=False):
        '''Continuous -- for single TX
           euclidian error when modeling the output representation is a matrix (image)
           both pred and truth are batches, typically a batch of 32
           now both prediction and truth are continuous numbers
        Args:
            pred_batch:  numpy.ndarray -- size=(N, 1, 100, 100)
            truth_batch: numpy.ndarray -- size=(N, num_tx, 2)
        Return:
            pred_locs -- list<np.ndarray>
            errors    -- list<list>
            misses    -- list
            false     -- list
        '''
        def float_target(pred_matrix, pred_peaks):
            new_pred_peaks = []
            for pred_x, pred_y in pred_peaks:
                sum_weight = 0
                neighbor = []
                for d in [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]:
                    nxt = (pred_x + d[0], pred_y + d[1])
                    if 0 <= nxt[0] < grid_len and 0 <= nxt[1] < grid_len:
                        neighbor.append(((nxt[0] + 0.5, nxt[1] + 0.5), pred[nxt[0]][nxt[1]]))
                        sum_weight += pred_matrix[nxt[0]][nxt[1]]
                pred_x, pred_y = 0, 0
                for n in neighbor:
                    loc = n[0]
                    w   = n[1]
                    pred_x += loc[0] / sum_weight * w
                    pred_y += loc[1] / sum_weight * w
                new_pred_peaks.append((pred_x, pred_y))
            return new_pred_peaks

        pred_locs, errors, misses, falses = [], [], [], []
        for i, pred, pred_n, truth, indx in zip(range(len(pred_batch)), pred_batch, pred_ntx, truth_batch, index):
            # 1: get the multiple predicted locations
            pred = pred[0]           # there is only one channel
            pred_peaks, _ = Utility.detect_peak(pred, np.round(pred_n, 0), peak_threshold)         # get the predictions  TIME: 23 milliseconds
            pred_peaks = float_target(pred, pred_peaks)
            pred_locs.append(pred_peaks)
            # 2: do a matching and get the error
            radius_threshold = Default.grid_length * Default.error_threshold
            error, miss, false = Utility.compute_error(pred_peaks, truth, radius_threshold, False)
            errors.append(error)
            misses.append(miss)
            falses.append(false)
            if debug:
                print(i, indx, 'pred', [(round(loc[0], 2), round(loc[1], 2)) for loc in pred_peaks], '; truth', \
                      [(round(loc[0], 2), round(loc[1], 2)) for loc in truth], ' ; error', error, ' ; miss', miss, ' ; false', false)
        return pred_locs, errors, misses, falses


    @staticmethod
    def localization_error_image_continuous_detection(pred_batch, truth_batch, index, debug=False):
        '''This is for the output of object detection, where there is no need to pass in num of TX. Because the output is directly (x, y)
           Continuous -- for single TX
           euclidian error when modeling the output representation is a matrix (image)
           both pred and truth are batches, typically a batch of 32
           now both prediction and truth are continuous numbers
        Args:
            pred_batch:  numpy.ndarray -- size=(N, num_tx, 2)
            truth_batch: numpy.ndarray -- size=(N, num_tx, 2)
        Return:
            pred_locs -- list<np.ndarray>
            errors    -- list<list>
            misses    -- list
            false     -- list
        '''
        pred_locs, errors, misses, falses = [], [], [], []
        for i, pred, truth, indx in zip(range(len(pred_batch)), pred_batch, truth_batch, index):
            # do a matching and get the error
            radius_threshold = Default.grid_length * Default.error_threshold
            error, miss, false = Utility.compute_error(pred, truth, radius_threshold, False)
            pred_locs.append(pred)
            errors.append(error)
            misses.append(miss)
            falses.append(false)
            if debug:
                print(i, indx, 'pred', [(round(loc[0], 2), round(loc[1], 2)) for loc in pred], '; truth', \
                      [(round(loc[0], 2), round(loc[1], 2)) for loc in truth], ' ; error', error, ' ; miss', miss, ' ; false', false)
        return pred_locs, errors, misses, falses


    @staticmethod
    def localization_error_image_continuous_detection_power(pred_batch, truth_batch, truth_power_batch, index, debug=False):
        '''Include the power estimation
           This is for the output of object detection, where there is no need to pass in num of TX. Because the output is directly (x, y)
           Continuous -- for single TX
           euclidian error when modeling the output representation is a matrix (image)
           both pred and truth are batches, typically a batch of 32
           now both prediction and truth are continuous numbers
        Args:
            pred_batch:  numpy.ndarray -- size=(N, num_tx, 2)
            truth_batch: numpy.ndarray -- size=(N, num_tx, 2)       ground truth for location
            truth_power_batch: np.ndarray -- size=(N, num_tx, 1)    ground truth for power
        Return:
            pred_locs -- list<np.ndarray>
            errors    -- list<list>
            misses    -- list
            false     -- list
        '''
        pred_locs, errors, misses, falses = [], [], [], []
        power_dicts = []
        for i, pred, truth, power, indx in zip(range(len(pred_batch)), pred_batch, truth_batch, truth_power_batch, index):
            # do a matching and get the error
            radius_threshold = Default.grid_length * Default.error_threshold
            error, miss, false, power_dict = Utility.compute_error_power(pred, truth, power, radius_threshold, False)
            pred_locs.append(pred)
            errors.append(error)
            misses.append(miss)
            falses.append(false)
            power_dicts.append(power_dict)
            if debug:
                print(i, indx, 'pred', [(round(loc[0], 2), round(loc[1], 2)) for loc in pred], '; truth', \
                      [(round(loc[0], 2), round(loc[1], 2)) for loc in truth], ' ; error', error, ' ; miss', miss, ' ; false', false, 'power dict', power_dict)
        return pred_locs, errors, misses, falses, power_dicts


    @staticmethod
    def loss(pred, y):
        n = len(pred) * len(pred[0])
        summ = np.sum((pred - y)**2)
        return summ/n

tf = T.Compose([
     UniformNormalize(Default.noise_floor),                 # TUNE: Uniform normalization is better than the above minmax normalization
     T.ToTensor()])



############# Below are for DeepTxFinder #################3


class SensorInputDatasetRegression(Dataset):
    '''Sensor reading input dataset -- for multi TX
       Output is image, model as a image segmentation problem
    '''
    def __init__(self, root_dir: str, grid_len:int, transform=None):
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
        target_name = str(idx%self.sample_per_label) + '.target.npy'
        target_arr = self.get_regression_target(folder, target_name)
        target_num = int(len(target_arr)/2)
        matrix = np.load(matrix_path)
        if self.transform:
            matrix = self.transform(matrix)
        target_arr = self.min_max_normalize(target_arr)
        sample = {'matrix': matrix, 'target': target_arr, 'target_num': target_num, 'index': idx}
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
        target = np.reshape(target, -1)
        return target.astype(np.float32)

    def min_max_normalize(self, target_arr: np.ndarray):
        '''scale the localization to a range of (0, 1)
        '''
        target_arr /= Default.grid_length
        return target_arr

    def undo_normalize(self, arr: np.ndarray):
        arr *= self.grid_len
        return arr


class Normalize:
    '''dtxf
    '''
    def __init__(self, noise_floor):
        self.noise_floor = noise_floor

    def __call__(self, matrix):
        matrix -= self.noise_floor
        max_value = max([max(l) for l in matrix])
        matrix /= (max_value)
        return matrix.astype(np.float32)

dtxf_tf = T.Compose([
     Normalize(Default.noise_floor),
     T.ToTensor()])

def match(pred, y):
    '''
    Args:
        pred: Tensor -- (batch size, num tx * 2)
        y   : Tensor -- (batch size, num tx * 2)
    '''
    batch_size = len(pred)
    dimension  = len(pred[0])
    pred_m, y_m = torch.zeros((batch_size, dimension)), torch.zeros((batch_size, dimension))
    for i in range(batch_size):
        pred2, y2 = [], []   # for one sample
        for j in range(0, dimension, 2):
            pred2.append(pred[i][j:j+2])
            y2.append(y[i][j:j+2])
        pred2, y2 = match_helper(pred2, y2)
        pred_m[i] = pred2
        y_m[i]    = y2
    return pred_m, y_m


def match_helper(pred, y):
    ''' up to now, assume len(pred) and len(y) is equal
        do a matching of the predictions and truth
    Args:
        pred: list<Tensor<float>> -- pairs of locations
        y   : list<Tensor> -- pairs of locations
    Return:
        Tensor<float>, Tensor<float>
    '''
    distances = np.zeros((len(pred), len(y)))
    for i in range(len(pred)):
        for j in range(len(y)):
            distances[i, j] = Utility.distance(pred[i], y[j])

    matches = []
    k = 0
    while k < min(len(pred), len(y)):
        min_error = np.min(distances)
        min_error_index = np.argmin(distances)
        i = min_error_index // len(y)
        j = min_error_index % len(y)
        matches.append((i, j, min_error))
        distances[i, :] = np.inf
        distances[:, j] = np.inf
        k += 1

    pred_m, y_m = [], []
    for i, j, e in matches:
        pred_m.append(pred[i])
        y_m.append(y[j])

    return torch.cat(pred_m), torch.cat(y_m)
