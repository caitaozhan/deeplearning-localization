'''
common utility
'''

import math
import os
import shutil
import numpy as np

class Utility:
    '''some common utilities'''
    @staticmethod
    def distance_propagation(indx2d_1: tuple, indx2d_2: tuple):
        '''euclidean distance for propagation model'''
        if indx2d_1 == indx2d_2:
            return 0.5
        else:
            return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)

    @staticmethod
    def distance(indx2d_1: tuple, indx2d_2: tuple):
        '''euclidean distance for localization error'''
        return math.sqrt((indx2d_1[0] - indx2d_2[0]) ** 2 + (indx2d_1[1] - indx2d_2[1]) ** 2)

    @staticmethod
    def remove_make(root_dir: str):
        '''if root_dir exists, remove all the content
           make directory
        '''
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.mkdir(root_dir)

    @staticmethod
    def db2linear(db: float):
        '''Transform power from decibel into linear format
        Args:
            db -- power in db
        Return:
            float -- power in linear form
        '''
        try:
            if db <=-80:
                return 0
            linear = np.power(10, db/10)
            return linear
        except Exception as e:
            print(e, db)

    @staticmethod
    def linear2db(linear: float):
        '''Transform power from linear into decibel format
        Args:
            linear -- power in linear
        Return:
            float -- power in decibel
        '''
        db = 10 * np.log10(linear)
        if db < -80 or db is np.nan:
            db = -80
        return db
