'''
Generate Receivers and training data
'''

from typing import List

import random
import numpy as np

from visualize import Visualize
from propagation import Propagation
from input_output import Default
from node import Sensor
from utility import Utility
import matplotlib.pyplot as plt


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
        self.propagation = Propagation(self.alpha, self.std)

    def generate(self, power: float, cell_percentage: float, sample_per_cell: int, sensor_file: str, output_dir: str):
        '''
        Args:
            power           -- the power of the transmitter
            cell_percentage -- percentage of cells being label (see if all discrete location needs to be labels)
            sample_per_cell -- samples per cell
            sensor_file -- sensor location file
            output_dir  -- the image output directory
        '''
        # 1 read the sensor file, do a checking
        if str(self.grid_length) not in sensor_file[:sensor_file.find('-')]:
            print(f'grid length {self.grid_length} and sensor file {sensor_file} not match')

        sensors = []
        with open(sensor_file, 'r') as f:
            indx = 0
            for line in f:
                x, y = line.split()
                sensors.append(Sensor(int(x), int(y), indx))
                indx += 1

        # 2 start from (0, 0), generate data, might skip some locations
        label_count = int(self.grid_length * self.grid_length * cell_percentage)
        population = [(i, j) for j in range(self.grid_length) for i in range(self.grid_length)]
        labels = random.sample(population, label_count)
        for label in labels:
            for i in range(sample_per_cell):
                tx = label
                grid = np.zeros((self.grid_length, self.grid_length))
                for sensor in sensors:
                    dist = Utility.distance(tx, (sensor.x, sensor.y))
                    pathloss = self.propagation.pathloss(dist)
                    grid[sensor.x][sensor.y] = power - pathloss
                plt.imsave(f'data/images/{tx}-{i}.png', grid)

        
if __name__ == '__main__':
    # GenerateSensors.random(100, 500, 0, 'data/sensors-100-500')
    gd = GenerateData(Default.random_seed, Default.alpha, Default.std, Default.grid_length, Default.cell_length, Default.sen_density)
    gd.generate(20, 1, 3, 'data/sensors/100-500', 'data/images/')
