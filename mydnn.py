'''
deep learning networks
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTranslation(nn.Module):
    '''Image translation.
       the first CNN is the same as NetTranslation, 
       the second one uses the output of the first CNN and output the # of Tx
       Assuming the input image is 1 x 100 x 100
    '''
    def __init__(self):
        super(NetTranslation, self).__init__()
        self.conv11 = nn.Conv2d(1, 8,  7, padding=3)   # TUNE: a larger filter decrease miss, decrease localization error
        self.conv12 = nn.Conv2d(8, 32, 7, padding=3)
        self.conv13 = nn.Conv2d(32, 1, 7, padding=3)

    def forward(self, x):
        # first CNN input is 1 x 100 x 100
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        y = self.conv13(x)
        return y

"""
In the two CNN in a sequence method. the second CNN that predicts # of TX will affect the output of the first CNN
--> two CNN in a sequence may be a bad idea
--> how about two CNN in parallel
"""
class NetNumTx(nn.Module):
    """this CNN predicts # of TX """
    def __init__(self, max_ntx):
        super(NetNumTx, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.conv3 = nn.Conv2d(4, 8, 5)
        self.groupnorm1 = nn.GroupNorm(1, 2)
        self.groupnorm2 = nn.GroupNorm(2, 4)
        self.groupnorm3 = nn.GroupNorm(4, 8)
        self.fc1 = nn.Linear(648, 32)
        self.fc2 = nn.Linear(32, max_ntx)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.groupnorm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.groupnorm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.groupnorm3(self.conv3(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetNumTx2(nn.Module):
    """this CNN predicts # of TX """

    def __init__(self, max_ntx):
        super(NetNumTx2, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv2 = nn.Conv2d(2, 4, 5, padding=2)
        self.conv3 = nn.Conv2d(4, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(8, 16, 5, padding=2)
        self.norm1 = nn.GroupNorm(1, 2)
        self.norm2 = nn.GroupNorm(2, 4)
        self.norm3 = nn.GroupNorm(4, 8)
        self.norm4 = nn.GroupNorm(8, 16)
        self.fc1 = nn.Linear(576, 50)
        self.fc2 = nn.Linear(50, max_ntx)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.norm3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.norm4(self.conv4(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetNumTx3(nn.Module):
    """this CNN predicts # of TX """

    def __init__(self, max_ntx):
        super(NetNumTx3, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, 2)
        self.norm2 = nn.GroupNorm(2, 4)
        self.norm3 = nn.GroupNorm(4, 8)
        self.norm4 = nn.GroupNorm(8, 16)
        self.norm5 = nn.GroupNorm(8, 16)
        self.fc1 = nn.Linear(144, 8)
        self.fc2 = nn.Linear(8, max_ntx)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.norm1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.norm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.norm3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.norm4(self.conv4(x))), 2)
        x = F.max_pool2d(F.relu(self.norm5(self.conv5(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetTranslation5(nn.Module):
    '''The chosen one! In the Jupyter Notebook, it's name is NetTranslation5_norm
       Image translation. Comparing with version 4, it adds a layer so it is symetric, it also has group normalization
       the first CNN is the same as NetTranslation, 
       the second one uses the output of the first CNN and output the # of Tx
       Assuming the input image is 1 x 100 x 100
    '''
    def __init__(self):
        super(NetTranslation5, self).__init__()
        self.conv1 = nn.Conv2d(1,  8,  5, padding=2)   # TUNE: a larger filter decrease miss, decrease localization error
        self.conv2 = nn.Conv2d(8,  32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 8,  5, padding=2)
        self.conv4 = nn.Conv2d(8,  1,  5, padding=2)
        self.norm1 = nn.GroupNorm(4,  8)
        self.norm2 = nn.GroupNorm(16, 32)
        self.norm3 = nn.GroupNorm(4,  8)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        y = self.conv4(x)
        return y



########## Below are for DeepTxFinder ###############

class CNN_NoTx(nn.Module):
    """this CNN predicts # of TX """
    def __init__(self, max_ntx):
        super(CNN_NoTx, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)
        self.dense = nn.Linear(8192, 512)
        self.dense1 = nn.Linear(512, max_ntx)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = self.flat(x)
        x = self.drop(x)
        x = F.relu(self.dense(x))
        x = self.dense1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_i(nn.Module):

    def __init__(self, ntx):
        super(CNN_i, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)
        self.dense = nn.Linear(8192, 512)
        self.dense1 = nn.Linear(512, 2*ntx)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = self.flat(x)
        x = self.drop(x)
        x = F.relu(self.dense(x))
        x = self.dense1(x)
        return x
