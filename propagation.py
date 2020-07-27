'''
A propagation model
log distance path loss + zero mean Gaussian shadowing
'''

import math
import numpy as np

class Propagation:
    '''Free space pathloss plus shadowing
    '''
    def __init__(self, alpha: float, std: float):
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
        freespace = 10 * math.log10(distance)
        shadowing = np.random.normal(loc=0, scale=self.std)
        return freespace + shadowing
