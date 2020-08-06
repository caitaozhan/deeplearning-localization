'''
common utility
'''

import math
import os
import shutil

class Utility:
    '''some common utilities'''
    @staticmethod
    def distance(indx2d_1, indx2d_2):
        '''euclidean distance'''
        return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)

    @staticmethod
    def remove_make(root_dir: str):
        '''if root_dir exists, remove all the content
           make directory
        '''
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.mkdir(root_dir)
