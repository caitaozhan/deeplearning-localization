import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import math
from skimage import io, transform
from torch import tensor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, '/home/caitao/Project/dl-localization')
from input_output import Default
from utility import Utility

from torch._six import container_abcs, string_classes, int_classes
import re
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")



# class SensorInputDatasetTranslation(Dataset):
#     '''Sensor reading input dataset -- for multi TX
#        Output is image, model as a image segmentation problem
#     '''
#     def __init__(self, root_dir: str, transform=None):
#         '''
#         Args:
#             root_dir:  directory with all the images
#             labels:    labels of images
#             transform: optional transform to be applied on a sample
#         '''
#         self.root_dir = root_dir
#         self.transform = transform
#         self.length = len(os.listdir(self.root_dir))
#         self.sample_per_label = self.get_sample_per_label()

#     def __len__(self):
#         return self.length * self.sample_per_label

#     def __getitem__(self, idx):
#         folder = int(idx/self.sample_per_label)
#         folder = format(folder, '06d')
#         matrix_name = str(idx%self.sample_per_label) + '.npy'
#         matrix_path = os.path.join(self.root_dir, folder, matrix_name)
#         target_name = str(idx%self.sample_per_label) + '.target.npy'
#         target_arr = self.get_regression_target(folder, target_name)
#         target_num = int(len(target_arr)/2)
#         matrix = np.load(matrix_path)
#         if self.transform:
#             matrix = self.transform(matrix)
#         target_arr = self.min_max_normalize(target_arr)
#         sample = {'matrix': matrix, 'target': target_arr, 'target_num': target_num, 'index': idx}
#         return sample

#     def get_sample_per_label(self):
#         folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
#         samples = glob.glob(os.path.join(folder, '*.npy'))
#         targets = glob.glob(os.path.join(folder, '*.target.npy'))
#         return len(samples) - len(targets)

#     def get_regression_target(self, folder: str, target_name: str):
#         '''
#         Args:
#             folder: example of folder is 000001
#         Return:
#             a two dimension matrix
#         '''
#         target_file = os.path.join(self.root_dir, folder, target_name)
#         target = np.load(target_file)
#         target = np.reshape(target, -1)
#         return target.astype(np.float32)

#     def min_max_normalize(self, target_arr: np.ndarray):
#         '''scale the localization to a range of (0, 1)
#         '''
#         target_arr /= Default.grid_length
#         return target_arr

#     def undo_normalize(self, arr: np.ndarray):
#         arr *= self.grid_len
#         return arr


class SensorInputDatasetTranslation(Dataset):
    '''Sensor reading input dataset -- for multi TX
       Output is image, model as a image segmentation problem
    '''
    def __init__(self, root_dir: str, transform=None, transform_pu=None):
        '''
        Args:
            root_dir:  directory with all the images
            labels:    labels of images
            transform: optional transform to be applied on a sample
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.transform_pu = transform_pu
        self.length = len(os.listdir(self.root_dir))
        self.sample_per_label = self.get_sample_per_label()

    def __len__(self):
        return self.length * self.sample_per_label

    def __getitem__(self, idx):
        folder = int(idx / self.sample_per_label)
        folder = format(folder, '06d')
        matrix_name = str(idx % self.sample_per_label) + '.npy'
        matrix_path = os.path.join(self.root_dir, folder, matrix_name)
        target_name = str(idx % self.sample_per_label) + '.target.npy'
        target_img, target_float = self.get_translation_target(folder, target_name)
        matrix = np.load(matrix_path)
        if self.transform:
            matrix = self.transform(matrix)
        target_num = len(target_float)
        power_name = str(idx % self.sample_per_label) + '.power.npy'
        power_path = os.path.join(self.root_dir, folder, power_name)
        power = np.load(power_path)
        matrix_auth_name = str(idx % self.sample_per_label) + '.auth.npy'
        matrix_auth_path = os.path.join(self.root_dir, folder, matrix_auth_name)
        matrix_auth = np.load(matrix_auth_path)
        if self.transform:
            matrix_auth = self.transform(matrix_auth)
        target_auth_float_name = str(idx % self.sample_per_label) + '.auth.target.npy'
        target_auth_float_path = os.path.join(self.root_dir, folder, target_auth_float_name)
        target_auth_float = np.load(target_auth_float_path)
        power_auth_name = str(idx % self.sample_per_label) + '.auth.power.npy'
        power_auth_path = os.path.join(self.root_dir, folder, power_auth_name)
        power_auth = np.load(power_auth_path)
        pu_matrix = self.get_pu_matrix(target_auth_float, power_auth)
        if self.transform_pu:
            pu_matrix = self.transform_pu(pu_matrix)
        two_sheet = np.stack((matrix_auth, pu_matrix), axis=1)
        sample = {'matrix': matrix, 'target': target_img, 'target_float': target_float, 'target_num': target_num, 'power': power, 'index': idx}
#         sample = {'matrix': matrix, 'target': target_img, 'target_float': target_float, 'target_num': target_num, 'power': power, 'index': idx, \
#                  'matrix_auth': matrix_auth, 'target_auth_float': target_auth_float, 'power_auth': power_auth, 'pu_matrix': pu_matrix, 'two_sheet': two_sheet}
        return sample

    def get_pu_matrix(self, target_auth_float, power_auth):
        '''The sheet for PU, each PU is a circle
        Args:
            target_auth_float -- np.ndarray, n=3 -- a list of 2D locations
            power_auth        -- np.ndarray, n=2 -- a list of power values
        Return:
            np.ndarray, n = 2
        '''
        grid = np.zeros((Default.grid_length, Default.grid_length))
        grid.fill(-2.5)  # PU power is between -2.5 and 2.5
        for (x, y), power in zip(target_auth_float, power_auth):
            x, y = int(x), int(y)
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    grid[x + i][y + j] = power + 1
        return grid.astype(np.float32)
    
    def get_sample_per_label(self):
        folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
        samples = glob.glob(os.path.join(folder, '*.json'))
        return len(samples)

    def gaussian(self, b_x, b_y, x, y):
        '''2D gaussian
        '''
        a, c = 10, 0.9
        dist = math.sqrt((x - b_x) ** 2 + (y - b_y) ** 2)
        return a * np.exp(- dist ** 2 / (2 * c ** 2))

    def get_translation_target(self, folder: str, target_name: str):
        '''try guassian peak
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
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    nxt = (x + i, y + j)
                    if 0 <= nxt[0] < Default.grid_length and 0 <= nxt[1] < Default.grid_length:
                        val = self.gaussian(target_float[0], target_float[1], nxt[0]+0.5, nxt[1]+0.5)
                        grid[nxt[0]][nxt[1]] = max(val, grid[nxt[0]][nxt[1]])
        grid = np.expand_dims(grid, 0)
        return grid.astype(np.float32), location.astype(np.float32)




class PuNormalize:
    '''only subtract the floor value
    '''
    
    def __init__(self, floor):
        self.floor = floor
    
    def __call__(self, matrix):
        matrix -= self.floor
        return matrix.astype(np.float32)


class UniformNormalize:
    '''Set a uniform threshold accross all samples
    '''
    def __init__(self, noise_floor):
        self.noise_floor = noise_floor

    def __call__(self, matrix):
        matrix -= self.noise_floor
        matrix /= (-self.noise_floor/2)
        return matrix.astype(np.float32)


tf = T.Compose([
     UniformNormalize(Default.noise_floor),                 # TUNE: Uniform normalization is better than the above minmax normalization
     T.ToTensor()])


tf_pu = T.Compose([
     PuNormalize(-2.5),
     T.ToTensor()])


def my_padding(batch, max_len):
    """add zeros to elements that are not maximum length"""
    for i in range(len(batch)):
        diff = max_len - len(batch[i])
        if diff > 0:                      # padding
            if len(batch[i].shape) == 1:
                zeros = torch.zeros(diff)
            elif len(batch[i].shape) == 2:
                zeros = torch.zeros(diff, 2)
            else:
                raise('unsupported dimension in my_padding')
            padded = torch.cat((batch[i], zeros), 0)
            batch[i] = padded


def my_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        max_len = len(max(batch, key=len))
        min_len = len(min(batch, key=len))
        if max_len != min_len:
            my_padding(batch, max_len)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        # max_len = len(max(batch, key=len))
        # min_len = len(min(batch, key=len))
        # if max_len != min_len:
        #     my_padding(batch, max_len)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(my_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


# root_dir = './data/205train'
# sensor_input_dataset = SensorInputDatasetTranslation(root_dir = root_dir, transform = tf)
# sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=my_collate)

root_dir = './data/1005train'
sensor_input_dataset = SensorInputDatasetTranslation(root_dir=root_dir, transform=tf, transform_pu=tf_pu)
sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=my_collate)

device = torch.device('cuda')


def my_uncollate(y_num, y_float):
    """this is for uncollating the target_float"""
    y_float_tmp = []
    for ntx, y_f in zip(y_num, y_float):
        y_float_tmp.append(y_f[:ntx*2])
    return np.array(y_float_tmp)


for t, sample in enumerate(sensor_input_dataloader):
    # X = sample['matrix'].to(device)
    # y = sample['target'].to(device)
    # y_num   = sample['target_num'].to(device)
    y_num2  = np.array(sample['target_num'])
    y = np.array(sample['target'])
    y = my_uncollate(y_num2, y)
    if t % 10 == 9:
        print(t, end=' ')
print('caitao')

