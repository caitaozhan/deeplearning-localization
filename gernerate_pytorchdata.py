from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import math
import glob
import json
import seaborn as sns
from torchviz import make_dot, make_dot_from_trace
from skimage import io, transform
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
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


class UniformNormalize:
    '''Set a uniform threshold accross all samples
    '''
    def __init__(self, noise_floor):
        self.noise_floor = noise_floor

    def __call__(self, matrix):
        matrix -= self.noise_floor
        matrix /= (-self.noise_floor/2)
        return matrix.astype(np.float32)


def my_padding(batch, max_len):
    """add zeros to elements that are not maximum length"""
    for i in range(len(batch)):
        diff = max_len - len(batch[i])
        if diff > 0:                      # padding
            zeros = torch.zeros(diff, 2)
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


def my_uncollate(y_num, y_float):
    """this is for uncollating the target_float"""
    y_float_tmp = []
    for ntx, y_f in zip(y_num, y_float):
        y_float_tmp.append(y_f[:ntx])
    return np.array(y_float_tmp, dtype=object)

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
        sample = {'matrix': matrix, 'target': target_img, 'target_float': target_float, 'target_num': target_num, 'index': idx}
        return sample

    def get_sample_per_label(self):
        folder = glob.glob(os.path.join(self.root_dir, '*'))[0]
        samples = glob.glob(os.path.join(folder, '*.json'))
        return len(samples)

    def get_translation_target_old(self, folder: str, target_name: str):
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

    def get_translation_target_old2(self, folder: str, target_name: str):
        '''9 pixels divide a total of 27, the middle pixel gets 9, the other 8 pixels get 18 by some weights.
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
            grid[x][y] = 9
            neighbor = []
            sum_weight = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    nxt = (x + i, y + j)
                    if 0 <= nxt[0] < Default.grid_length and 0 <= nxt[1] < Default.grid_length:
                        weight = 1./Utility.distance((nxt[0] + 0.5, nxt[1] + 0.5), target_float)
                        sum_weight += weight
                        neighbor.append((nxt, weight))
            for n, w in neighbor:
                grid[n[0]][n[1]] += w / sum_weight * 18
        grid = np.expand_dims(grid, 0)
        return grid.astype(np.float32), location.astype(np.float32)


    def get_translation_target_8sum9(self, folder: str, target_name: str):
        '''the sum of the 8 surrounding pixels is 9
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
            grid[x][y] = 9
            neighbor = []
            sum_weight = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    nxt = (x + i, y + j)
                    if 0 <= nxt[0] < Default.grid_length and 0 <= nxt[1] < Default.grid_length:
                        weight = 1./Utility.distance((nxt[0] + 0.5, nxt[1] + 0.5), target_float)
                        sum_weight += weight
                        neighbor.append((nxt, weight))
            for n, w in neighbor:
                grid[n[0]][n[1]] += w / sum_weight * 9
        grid = np.expand_dims(grid, 0)
        return grid.astype(np.float32), location.astype(np.float32)  

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


    def save_object_detection_image(self, output_dir, pred_matrix, idx):
        '''Take a 2D matrix (one channel image) as input, output as an image with three channels (by duplication)
        Args:
            pred_matrix -- np.ndarray, n=2
        '''
        img = np.stack((pred_matrix, pred_matrix, pred_matrix), axis=2)
        filename = format(idx, '06d')
        path = os.path.join(output_dir, filename)
        np.save(path, img.astype(np.float32))
        return filename

    def save_object_detection_label(self, output_dir, idx, grid_len, pred_matrix, peak_threashold=0.3, debug=False):
        filename = format(idx, '06d') + '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            sample = self.__getitem__(idx)
            ground_truth = sample['target_float']
            for x, y in ground_truth:
                if debug:
                    print(x, y)
                if pred_matrix[int(x)][int(y)] > peak_threashold:
                    f.write('0 {} {} {} {}\n'.format(y / grid_len, x / grid_len, 5 / grid_len, 5 / grid_len))
                else:
                    print(f'index = {idx} no peak at ({x}, {y}), value is {pred_matrix[int(x)][int(y)]}')


tf = T.Compose([
     UniformNormalize(Default.noise_floor),                 # TUNE: Uniform normalization is better than the above minmax normalization
     T.ToTensor()])

# training
i = 30006
root_dir = './data/1016train'
sensor_input_dataset = SensorInputDatasetTranslation(root_dir=root_dir, transform=tf)
sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=32, shuffle=True, num_workers=3, collate_fn=my_collate)
print('matrix shape:', sensor_input_dataset[i]['matrix'].shape)
print('matrix shape:', sensor_input_dataset[i]['matrix'])
print('target:', sensor_input_dataset[i]['target'])
print('target float:', sensor_input_dataset[i]['target_float'])
print('length:', sensor_input_dataset.__len__())
print('label per sample:', sensor_input_dataset.get_sample_per_label())

fig, ax = plt.subplots(1, figsize=(5, 4))
sns.heatmap(sensor_input_dataset[i]['target'][0][0:5, 0:5], annot=True)


print('---\n')
# testing
root_dir = './data/1016test'
sensor_input_test_dataset = SensorInputDatasetTranslation(root_dir=root_dir, transform=tf)
sensor_input_test_dataloader = DataLoader(sensor_input_test_dataset, batch_size=32, shuffle=True, num_workers=3, collate_fn=my_collate)
print('matrix type:', sensor_input_test_dataset[i]['matrix'].dtype)
print('target type:', sensor_input_test_dataset[i]['target'].dtype)
print('target float:', sensor_input_test_dataset[i]['target_float'])
print('target num:', sensor_input_test_dataset[i]['target_num'])
print(sensor_input_test_dataset.__len__())


root_dir = './data/1016train'
sensor_input_dataset = SensorInputDatasetTranslation(root_dir=root_dir, transform=tf)
sensor_input_dataloader = DataLoader(sensor_input_dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=my_collate)

images_dir = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/images/train'
labels_dir = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/labels/train'
list_path  = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/train.txt'

device = torch.device('cuda')
# path5 = 'model/model1-12.13-net5-norm-32-splat.pt'
path5 = 'model/model1-12.18-net5-norm-32-splat.pt'
model5_f5_norm = NetTranslation5_norm()
model5_f5_norm.load_state_dict(torch.load(path5))
model5_f5_norm = model5_f5_norm.to(device)
model5_f5_norm.eval()


with open(list_path, 'w') as f:
    for t, sample in enumerate(sensor_input_dataloader):
        if t % 10 == 9:
            print(t, end=' ')
        X = sample['matrix'].to(device)
        indx = sample['index']
        pred_matrix = model5_f5_norm(X)
        for idx, matrix in zip(indx, pred_matrix):
            idx = idx.data.cpu().numpy()
            matrix = matrix.data.cpu().numpy()
            image_name = sensor_input_dataset.save_object_detection_image(images_dir, matrix[0], idx)
            sensor_input_dataset.save_object_detection_label(labels_dir, idx, Default.grid_length, matrix[0])
            path = os.path.join(images_dir, image_name) + '.npy'
            f.write(path + '\n')


# testing or valid

root_dir = './data/1016test'
sensor_input_test_dataset = SensorInputDatasetTranslation(root_dir=root_dir, transform=tf)
sensor_input_test_dataloader = DataLoader(sensor_input_test_dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=my_collate)

images_dir  = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/images/val'
labels_dir  = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/labels/val'
list_path   = '/home/caitao/Project/PyTorch-YOLOv3/data/deepmtl-1016/valid.txt'

with open(list_path, 'w') as f:
    for t, sample in enumerate(sensor_input_test_dataloader):
        if t % 10 == 9:
            print(t, end=' ')
        X = sample['matrix'].to(device)
        indx = sample['index']
        pred_matrix = model5_f5_norm(X)
        for idx, matrix in zip(indx, pred_matrix):
            idx = idx.data.cpu().numpy()
            matrix = matrix.data.cpu().numpy()
            image_name = sensor_input_test_dataset.save_object_detection_image(images_dir, matrix[0], idx)
            sensor_input_test_dataset.save_object_detection_label(labels_dir, idx, Default.grid_length, matrix[0])
            path = os.path.join(images_dir, image_name) + '.npy'
            f.write(path + '\n')