'''
visualization
'''

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Visualize:
    '''some visualization
    '''
    @classmethod
    def sensors(cls, sensors: List[int], grid_length: int, fig: int):
        '''visualize the sensor location
        '''
        grid = np.zeros((grid_length, grid_length))
        for sensor in sensors:
            x = sensor // grid_length
            y = sensor % grid_length
            grid[x][y] = 1
        sns.set(style="white")
        plt.subplots(figsize=(25, 25))
        sns.heatmap(grid, center=0, square=True, linewidth=1, cbar_kws={"shrink": .5}, annot=False)
        plt.savefig(f'visualize/{fig}')
