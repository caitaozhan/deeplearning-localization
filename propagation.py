'''
A propagation model
log distance path loss + zero mean Gaussian shadowing
'''

import math
import json
import numpy as np
from input_output import Default


class Propagation:
    '''Free space pathloss plus shadowing
    '''
    def __init__(self, alpha: float = Default.alpha, std: float = Default.std):
        '''
        Args:
            alpha -- the pathloss exponent
            std   -- the standard deviation of the zero mean Gaussian shadowing
        '''
        self.alpha = alpha
        self.std   = std

    def pathloss(self, distance: float):
        '''
        Args:
            distance -- the distance between two locations
        Return:
            pathloss: float
        '''
        freespace = 10 * self.alpha * math.log10(distance) if distance > 1 else 0
        shadowing = np.random.normal(0, self.std)
        pathloss = freespace + shadowing
        return pathloss if pathloss > 0 else -pathloss

class Splat:
    '''splat data
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load()

    def load(self):
        print('loading data ...')
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        print('finish loading')
        return np.asarray(data)
    
    def pathloss(self, x1: float, y1: float, x2: float, y2: float):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        return self.data[x1, y1, x2, y2]



def test():
    p = Propagation(2, 1)
    print(p.pathloss(0))
    print(p.pathloss(3))
    print(p.pathloss(15))

def test2():
    splat = Splat('data_splat/pl_map_array.json')
    x1, y1, x2, y2 = 1, 2, 1, 2
    print(f'({x1}, {y1}) --> ({x2}, {y2}) = {splat.pathloss(x1, y1, x2, y2)}')
    x1, y1, x2, y2 = 1, 2, 5, 6
    print(f'({x1}, {y1}) --> ({x2}, {y2}) = {splat.pathloss(x1, y1, x2, y2)}')
    x1, y1, x2, y2 = 1, 2, 20, 21
    print(f'({x1}, {y1}) --> ({x2}, {y2}) = {splat.pathloss(x1, y1, x2, y2)}')
    x1, y1, x2, y2 = 0, 0, 30, 0
    print(f'({x1}, {y1}) --> ({x2}, {y2}) = {splat.pathloss(x1, y1, x2, y2)}')
    import time
    time.sleep(10)

if __name__ == '__main__':
    # test()
    test2()
