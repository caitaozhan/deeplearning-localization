'''
my deep learning networks
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    '''The output dimension of the full connnection layer is 100 x 100 = 10000
    '''
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 32, 5)
        self.fc    = nn.Linear(15488, 10000)  # too many labels

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    net = Net1()
    print(net)
    inputt = torch.randn(1, 1, 100, 100)
    output = net(inputt)
    print(output)
