'''
common utility
'''

import math

class Utility:
    @staticmethod
    def distance(indx2d_1, indx2d_2):
        '''euclidean distance'''
        return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)
