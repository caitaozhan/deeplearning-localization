'''
deep learning networks
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


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



class PowerPredictor5(nn.Module):
    '''The input is 1 x 21 x 21, the output is a scaler between 0.5 and 5.5
       No fully connected layer in the end
       
       The CHOSEN one.
    '''
    def __init__(self):
        super(PowerPredictor5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.conv3 = nn.Conv2d(128, 32, 5)
        self.conv4 = nn.Conv2d(32, 8, 5)
        self.conv5 = nn.Conv2d(8, 1, 5)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(32)
        self.norm4 = nn.BatchNorm2d(8)
    
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class SubtractNet3(nn.Module):
    '''The chosen one! Simply adding layers helps
    '''
    def __init__(self):
        super(SubtractNet3, self).__init__()
        self.conv1 = nn.Conv2d(2,  8,  5, padding=2)   # TUNE: a larger filter decrease miss, decrease localization error
        self.conv2 = nn.Conv2d(8,  16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv5 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv6 = nn.Conv2d(16, 8,  5, padding=2)
        self.conv7 = nn.Conv2d(8,  2,  5, padding=2)
        self.conv8 = nn.Conv2d(2,  1,  5, padding=2)
        self.norm1 = nn.GroupNorm(4,  8)
        self.norm2 = nn.GroupNorm(8,  16)
        self.norm3 = nn.GroupNorm(16, 32)
        self.norm4 = nn.GroupNorm(16, 32)
        self.norm5 = nn.GroupNorm(8, 16)
        self.norm6 = nn.GroupNorm(4,  8)
        self.norm7 = nn.GroupNorm(1,  2)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.norm6(self.conv6(x)))
        x = F.relu(self.norm7(self.conv7(x)))
        y = self.conv8(x)
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

