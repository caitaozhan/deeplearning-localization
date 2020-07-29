'''
A propagation model
log distance path loss + zero mean Gaussian shadowing
'''

import math
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



class GenerateData:
    '''Generate synthetic data from the propagation model
    '''
    def __init__(self, random_seed, alpha, std, grid_length, cell_length):
        self.random_seed = random_seed
        self.alpha = alpha
        self.std = std
        self.grid_length = grid_length
        self.cell_length = cell_length



def test():
    p = Propagation(2, 1)
    print(p.pathloss(0))
    print(p.pathloss(3))
    print(p.pathloss(15))


if __name__ == '__main__':
    test()
