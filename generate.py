'''
Generate Receivers and training data
'''

from typing import List

import random
import numpy as np

from visualize import Visualize


class GenerateSensors:
    '''generate sensors
    '''
    @classmethod
    def random(cls, grid_length: int, sensor_density: int, seed: int, filename: str):
        '''randomly generate some sensors in a grid
        '''
        random.seed(seed)
        all_sensors = list(range(grid_length * grid_length))
        subset_sensors = random.sample(all_sensors, sensor_density)
        Visualize.sensors(subset_sensors, grid_length, 1)
        subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length)
        Visualize.sensors(subset_sensors, grid_length, 2)
        subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length)
        Visualize.sensors(subset_sensors, grid_length, 3)
        subset_sensors.sort()
        GenerateSensors.save(subset_sensors, grid_length, filename)

    @classmethod
    def relocate_sensors(cls, random_sensors: List, grid_len: int):
        '''Relocate sensors that are side by side
        '''
        new_random_sensors = []
        need_to_relocate = []
        ocupy_grid = np.zeros((grid_len, grid_len), dtype=int)
        random_sensors.sort()
        for sen in random_sensors:
            s_x = sen // grid_len
            s_y = sen % grid_len
            if ocupy_grid[s_x][s_y] == 1:
                need_to_relocate.append(sen)
            else:
                new_random_sensors.append(sen)
                for x, y in [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]:
                    try:
                        ocupy_grid[s_x + x][s_y + y] = 1
                    except:
                        pass
        available = []
        for x in range(grid_len):
            for y in range(grid_len):
                if ocupy_grid[x][y] == 0:
                    available.append(x*grid_len + y)

        relocated = random.sample(available, len(need_to_relocate))
        new_random_sensors.extend(relocated)
        return new_random_sensors

    @classmethod
    def save(cls, sensors: List[int], grid_length: int, filename: str):
        with open(filename, 'w') as f:
            for sensor in sensors:
                x = sensor // grid_length
                y = sensor % grid_length
                f.write(f'{x} {y}\n')


class GenerateData:
    '''generate training data using a propagation model
    '''
    def __init__(self, seed: int, alpha: float, std: float, grid_length: int, cell_length: int, sensor_density: int):
        self.seed = seed
        self.alpha = alpha
        self.std = std
        self.grid_length = grid_length
        self.cell_length = cell_length
        self.sensor_density = sensor_density


if __name__ == '__main__':
    GenerateSensors.random(100, 500, 0, 'data/sensors')
