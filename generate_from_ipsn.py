'''
Generate realistic data from the ISPN 2020 testbed
'''

from dataclasses import dataclass
import os
import numpy as np
import random
from utility import Utility

@dataclass
class Sensor:
    '''class for a sensor
    '''
    x: int
    y: int
    std: float
    index: int


class IPSN:
    '''Generate training/testing data from the IPSN 2020 testbed
    '''
    def __init__(self, deepmtl_grid_len: int, ipsn_grid_len: int, ipsn_data_dir: str, ipsn_noise_floor: float):
        '''
        Args:
            deepmtl_grid_len -- grid length of the deepmtl dataset
            grid_len -- grid length of the IPSN dataset
            data_dir -- the directory for the IPSN dataset
            ipsn_noise_floor -- the noise floor in the IPSN testbed
        '''
        self.deepmtl_grid_len = deepmtl_grid_len
        self.ipsn_grid_len = ipsn_grid_len
        self.ipsn_data_dir = ipsn_data_dir
        self.ipsn_noise_floor = ipsn_noise_floor
        self.sensors = []
        self.means = []
        self.stds = []

    def init_data(self):
        '''Initialize the IPSN testbed data (the hypothesis file and sensors data)
        '''
        self.sensors = []
        sensor_file = os.path.join(self.ipsn_data_dir, 'sensors')
        index = 0
        with open(sensor_file, 'r') as f:
            for line in f:
                line = line.split(' ')
                x, y, std = int(line[0]), int(line[1]), float(line[2])
                self.sensors.append(Sensor(x, y, std, index))
                index += 1

        self.means = np.zeros((self.ipsn_grid_len * self.ipsn_grid_len, len(self.sensors)))
        self.stds = np.zeros((self.ipsn_grid_len * self.ipsn_grid_len, len(self.sensors)))
        hypothesis_file = os.path.join(self.ipsn_data_dir, 'hypothesis')
        with open(hypothesis_file, 'r') as f:
            count = 0
            for line in f:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                mean, std = float(line[4]), float(line[5])
                self.means[tran_x*self.ipsn_grid_len + tran_y, count] = mean # count equals to the index of the sensor
                self.stds[tran_x*self.ipsn_grid_len + tran_y, count] = std
                count = (count + 1) % len(self.sensors)

    def check_if_skip(self, small_x, small_y):
        '''Due to edges and finer granularity, some locations will be skipped or ignored
           The IPSN testbed data is 10x10, scale it to 20x20, then duplicate 25 of them and get a 100x100 grid
           So, first map the location from the 100x100 big grid to the 20x20 small grid
           then, see if the fine granularity location in the 20x20 grid is in the coarse granularity location in the 10x10 grid
        Args:
            small_x -- int -- the x location in the 20x20 grid
            small_y -- int -- the y location in the 20x20 grid
        Return:
            bool
        '''
        if small_x <= 3 or small_x >= 16 or small_y <= 3 or small_y >= 16:   # at the edge
            return True
        if small_x % 2 == 0 and small_y % 2 == 0:
            return False
        else:
            return True
    
    def linear2db(self, linear: float):
        '''Transform power from linear into decibel format
        Args:
            linear -- power in linear
        Return:
            float -- power in decibel
        '''
        if linear == 0:
            return self.ipsn_noise_floor
        db = 10 * np.log10(linear)
        if db < self.ipsn_noise_floor:
            db = self.ipsn_noise_floor
        return db

    def db2linear(self, db: float):
        try:
            if db <= self.ipsn_noise_floor:
                return 0
            linear = np.power(10, db/10)
            return linear
        except Exception as e:
            print(e, db)

    def add_smallgrid_to_biggrid(self, smallgrid, biggrid, x_low, x_high, y_low, y_high):
        '''Add the RSSI 20x20 small grid to the 100x100 big grid
           x_low, x_high, y_low, y_high specifies the locations
        '''
        for x in range(x_low, x_high):
            for y in range(y_low, y_high):
                i = x - x_low
                j = y - y_low
                biggrid[x, y] = self.linear2db(self.db2linear(biggrid[x, y]) + self.db2linear(smallgrid[i, j]))

    def generate_data(self, root_dir: str, sample_per_label: int, num_tx: int):
        '''Generate the training/testing data for the DeepMTL
        Args:
            root_dir -- the output directory
            sample_per_label -- the number of samples per label (a location)
            num_tx -- number of transmitters (the upper bound)
        '''
        Utility.remove_make(root_dir)
        population = []  # all the labels (TX location)
        for i in range(self.deepmtl_grid_len*self.deepmtl_grid_len):
            big_x = i // self.deepmtl_grid_len  # big: in the 100x100 grid
            big_y = i % self.deepmtl_grid_len   # small: in the 20x20 grid
            small_x = big_x % 20
            small_y = big_y % 20
            if self.check_if_skip(small_x, small_y):
                continue
            population.append((big_x, big_y))

        counter = 0  # overall counter
        for big_x, big_y in population:  # the current label (big_x, big_y)
            folder = f'{root_dir}/{counter:06d}'
            os.mkdir(folder)
            
            for i in range(sample_per_label):
                targets = []
                grid_big = np.zeros((self.deepmtl_grid_len, self.deepmtl_grid_len))
                grid_big.fill(self.ipsn_noise_floor)

                # step 1: get all the TX locations
                txs = [(big_x, big_y)]
                num_tx_cur = random.randint(1, num_tx)
                if num_tx_cur > 1:
                    population_set = set(population)
                    population_set.remove((big_x, big_y))
                    tmp_txs = random.sample(population_set, num_tx_cur-1)
                    txs.extend(tmp_txs)

                for big_x_, big_y_ in txs:
                    # step 2: fill the 20x20 small grid
                    grid_small = np.zeros((self.ipsn_grid_len*2, self.ipsn_grid_len*2)) # the scale factor for the ipsn testbed data is 2
                    grid_small.fill(self.ipsn_noise_floor)
                    small_x = big_x_ % 20
                    small_y = big_y_ % 20
                    ipsn_x, ipsn_y = small_x // 2, small_y // 2
                    for sensor in self.sensors:
                        mean = self.means[ipsn_x * self.ipsn_grid_len + ipsn_y, sensor.index]
                        std  = self.stds[ipsn_x * self.ipsn_grid_len + ipsn_y, sensor.index]
                        rssi = np.random.normal(mean, std, size=1)[0]
                        grid_small[sensor.x*2, sensor.y*2] = rssi if rssi > self.ipsn_noise_floor else self.ipsn_noise_floor

                    # step 3: add the 20x20 small grid to the 100x100 big grid
                    x_multiple = big_x_ // 20
                    y_multiple = big_y_ // 20
                    x_low, x_high = x_multiple * 20, (x_multiple + 1) * 20
                    y_low, y_high = y_multiple * 20, (y_multiple + 1) * 20
                    self.add_smallgrid_to_biggrid(grid_small, grid_big, x_low, x_high, y_low, y_high)

                    x_float, y_float = big_x_ + random.uniform(0, 1), big_y_ + random.uniform(0, 1)
                    targets.append([x_float, y_float])
                np.save(f'{folder}/{i}.npy', grid_big.astype(np.float32))
                np.save(f'{folder}/{i}.target.npy', np.array(targets).astype(np.float32))
            counter += 1


if __name__ == '__main__':
    np.random.seed(100)
    random.seed(100)
    deepmtl_grid_len = 100
    ipsn_grid_len = 10
    ipsn_data_dir = 'ipsn_testbed'
    ipsn_noise_floor = -48.5
    ipsn = IPSN(deepmtl_grid_len, ipsn_grid_len, ipsn_data_dir, ipsn_noise_floor)
    ipsn.init_data()

    root_dir = 'ipsn_testbed/test'
    sample_per_label = 10
    num_tx = 10
    ipsn.generate_data(root_dir, sample_per_label, num_tx)
