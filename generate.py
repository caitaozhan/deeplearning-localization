'''
Generate sensors and training data
'''

from typing import List

import random
import numpy as np
import imageio
import argparse
import os
import time
from visualize import Visualize
from propagation import Propagation, Splat
from input_output import Default, IpsnInput
from node import Sensor
from utility import Utility


class GenerateSensors:
    '''generate sensors
    '''
    @classmethod
    def random(cls, grid_length: int, sensor_density: int, seed: int, filename: str):
        '''randomly generate some sensors in a grid
        '''
        print('grid_length', grid_length)
        print('sensor density', sensor_density)
        print('seed', seed)
        print('filename', filename)
        random.seed(seed)
        all_sensors = list(range(grid_length * grid_length))
        subset_sensors = random.sample(all_sensors, sensor_density)
        Visualize.sensors(subset_sensors, grid_length, 1)
        subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length, sensor_density)
        Visualize.sensors(subset_sensors, grid_length, 2)
        # subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length, sensor_density)
        # Visualize.sensors(subset_sensors, grid_length, 3)
        # subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length, sensor_density)
        # Visualize.sensors(subset_sensors, grid_length, 4)
        # subset_sensors = GenerateSensors.relocate_sensors(subset_sensors, grid_length, sensor_density)
        # Visualize.sensors(subset_sensors, grid_length, 5)
        subset_sensors.sort()
        GenerateSensors.save(subset_sensors, grid_length, filename)

    @classmethod
    def relocate_sensors(cls, random_sensors: List, grid_len: int, sensor_density: int):
        '''Relocate sensors that are side by side
        '''
        if sensor_density <= 100:
            neighbor = [(i, j) for i in range(-5, 6) for j in range(-5, 6)]
        elif sensor_density <= 200:
            neighbor = [(i, j) for i in range(-4, 5) for j in range(-4, 5)]
        elif sensor_density <= 400:
            neighbor = [(i, j) for i in range(-3, 3) for j in range(-3, 3)]
        elif sensor_density <= 600:
            neighbor = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
        elif sensor_density <= 800:
            neighbor = [(i, j) for i in range(-2, 2) for j in range(-2, 3)]
        elif sensor_density <= 1000:
            neighbor = [(i, j) for i in range(-2, 2) for j in range(-2, 2)]
        else:
            pass
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
                for x, y in neighbor:
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
    def __init__(self, seed: int, alpha: float, std: float, grid_length: int, cell_length: int, sensor_density: int, noise_floor: int):
        self.seed = seed
        self.alpha = alpha
        self.std = std
        self.grid_length = grid_length
        self.cell_length = cell_length
        self.sensor_density = sensor_density
        self.noise_floor = noise_floor
        self.propagation = Propagation(self.alpha, self.std)

    def log(self, power, cell_percentage, sample_per_label, sensor_file, root_dir, num_tx, num_tx_upper, min_dist, max_dist, edge, splat):
        '''the meta data of the data
        '''
        with open(root_dir + '.txt', 'w') as f:
            f.write(f'seed              = {self.seed}\n')
            f.write(f'alpha             = {self.alpha}\n')
            f.write(f'std               = {self.std}\n')
            f.write(f'grid length       = {self.grid_length}\n')
            f.write(f'cell length       = {self.cell_length}\n')
            f.write(f'sensor density    = {self.sensor_density}\n')
            f.write(f'noise floor       = {self.noise_floor}\n')
            f.write(f'power             = {power}\n')
            f.write(f'cell percentage   = {cell_percentage}\n')
            f.write(f'sample per label  = {sample_per_label}\n')
            f.write(f'sensor file       = {sensor_file}\n')
            f.write(f'root file         = {root_dir}\n')
            f.write(f'number of TX      = {num_tx}\n')
            f.write(f'num TX upperbound = {num_tx_upper}\n')
            f.write(f'min distance      = {min_dist}\n')
            f.write(f'max distance      = {max_dist}\n')
            f.write(f'edge              = {edge}\n')
            f.write(f'splat             = {splat}\n')

    def check_edge(self, tx, edge):
        '''if tx is at the edge, return false
        Args:
            tx -- tuple<int, int>
        '''
        if edge <= tx[0] < self.grid_length-edge and edge <= tx[1] < self.grid_length-edge:
            return True
        return False

    def generate(self, power: float, cell_percentage: float, sample_per_label: int, sensor_file: str,\
                 root_dir: str, num_tx: int, num_tx_upper: bool, min_dist: int, max_dist: int, edge: int, vary_power: int, splat: bool):
        '''
        The generated input data is not images, but instead matrix. Because saving as images will loss some accuracy
        Args:
            power            -- the power of the transmitter
            cell_percentage  -- percentage of cells being label (see if all discrete location needs to be labels)
            sample_per_label -- samples per cell
            sensor_file      -- sensor location file
            root_dir         -- the output directory
        '''
        Utility.remove_make(root_dir)
        self.log(power, cell_percentage, sample_per_label, sensor_file, root_dir, num_tx, num_tx_upper, min_dist, max_dist, edge, splat)
        # random.seed(self.seed)   # need to comment this line when running simulate_data.py, or they will generate the same TX locations
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
        population = [(i, j) for i in range(self.grid_length) for j in range(self.grid_length) if self.check_edge((i, j), edge)]  # Tx is not at edge
        label_count = int(len(population)*cell_percentage)
        labels = random.sample(population, label_count)

        counter = 0
        for label in sorted(labels):
            tx = label           # each label create a directory
            tx_float = (tx[0] + random.uniform(0, 1), tx[1] + random.uniform(0, 1))
            if counter % 100 == 0:
                print(f'{counter/len(labels)*100}%')
            folder = f'{root_dir}/{counter:06d}'
            os.mkdir(folder)     # update on Aug. 27, change the name of the folder from label to counter index
            for i in range(sample_per_label):
                power_delta  = random.uniform(-vary_power, vary_power)
                targets      = [tx_float]          # ground truth location
                power_deltas = [power_delta]       # ground truth power
                powers       = [power + power_delta]
                grid = np.zeros((self.grid_length, self.grid_length))
                grid.fill(Default.noise_floor)
                for sensor in sensors:
                    if not splat:
                        dist = Utility.distance_propagation(tx_float, (sensor.x, sensor.y)) * Default.cell_length
                        pathloss = self.propagation.pathloss(dist)
                    else:
                        pathloss = mysplat.pathloss(tx_float[0], tx_float[1], sensor.x, sensor.y) + random.uniform(-0.25, 0.25)
                    rssi = (power + power_delta) - pathloss
                    grid[sensor.x][sensor.y] = rssi if rssi > Default.noise_floor else Default.noise_floor
                # the other TX
                population_set = set(population)
                if num_tx_upper is False:
                    num_tx_copy = num_tx
                else:
                    num_tx_copy = random.randint(1, num_tx)
                intru = tx
                while num_tx_copy > 1:   # get one new TX at a time
                    self.update_population(population_set, intru, Default.grid_length, min_dist, max_dist)
                    ntx = random.sample(population_set, 1)[0]
                    ntx = (ntx[0] + random.uniform(0, 1), ntx[1] + random.uniform(0, 1))  # TX is not at the center of grid cell
                    power_delta = random.uniform(-vary_power, vary_power)
                    targets.append(ntx)
                    power_deltas.append(power_delta)
                    powers.append(power + power_delta)
                    for sensor in sensors:
                        if not splat:
                            dist = Utility.distance_propagation(ntx, (sensor.x, sensor.y)) * Default.cell_length
                            pathloss = self.propagation.pathloss(dist)
                        else:
                            pathloss = mysplat.pathloss(ntx[0], ntx[1], sensor.x, sensor.y) + random.uniform(-0.25, 0.25)
                        rssi = (power + power_delta) - pathloss
                        exist_rssi = grid[sensor.x][sensor.y]
                        grid[sensor.x][sensor.y] = Utility.linear2db(Utility.db2linear(exist_rssi) + Utility.db2linear(rssi))
                    num_tx_copy -= 1
                    intru = ntx
                np.save(f'{folder}/{i}.npy', grid.astype(np.float32))
                np.save(f'{folder}/{i}.target', np.array(targets).astype(np.float32))
                np.save(f'{folder}/{i}.power', np.array(powers).astype(np.float32))  # Dec. 12, 2020: save in txt format (currently not used in the CNN)
                self.save_ipsn_input(grid, targets, power_deltas, sensors, f'{folder}/{i}.json')
                # if i == 0:
                #     imageio.imwrite(f'{folder}/{tx}.png', grid)
            counter += 1


    def save_ipsn_input(self, grid, targets, power_deltas, sensors, output_file):
        '''
        Args:
            grid         -- np.ndarray, n = 2
            targets      -- list<tuple<float, float>>
            power_deltas -- list<float>                        -- power of the TX, varying
            sensors      -- list<Sensors>
            output_file  -- str
        '''
        tx_data = {}
        for i, tx in enumerate(targets):
            tx_data[str(i)] = {
                "location": (round(tx[0], 3), round(tx[1], 3)),
                "gain": round(power_deltas[i], 3)
            }
        sensor_data = {}
        for i, sen in enumerate(sensors):
            sensor_data[str(i)] = round(grid[sen.x][sen.y], 3)
        ipsn_input = IpsnInput(tx_data, sensor_data)
        with open(output_file, 'w') as f:
            f.write(ipsn_input.to_json_str())


    def generate_ipsn(self, power: str, sensor_file: str, root_dir: str, splat: bool):
        '''generate the training data for the IPSN20 localization algorithm
        '''
        # sensors
        Utility.remove_make(root_dir)
        sensors = []
        with open(sensor_file, 'r') as f, open(root_dir + '/sensors', 'w') as f_out:
            indx = 0
            for line in f:
                x, y = line.split()
                f_out.write('{} {} {} {}\n'.format(x, y, 1, 1))  # uniform cost
                sensors.append(Sensor(int(x), int(y), indx))
                indx += 1

        # covariance matrix
        sen_num = len(sensors)
        with open(root_dir + '/cov', 'w') as f:
            cov = np.zeros((sen_num, sen_num))
            for i in range(sen_num):
                for j in range(sen_num):
                    if i == j:
                        cov[i, j] = 1   # assume the std is 1
                    f.write('{} '.format(cov[i, j]))
                f.write('\n')

        # hypothesis
        total_loc = self.grid_length*self.grid_length
        with open(root_dir + '/hypothesis', 'w') as f:
            for tx_1dindex in range(total_loc):
                t_x = tx_1dindex // self.grid_length
                t_y = tx_1dindex % self.grid_length
                for sen in sensors:
                    if not splat:
                        dist = Utility.distance_propagation((t_x, t_y), (sen.x, sen.y)) * Default.cell_length
                        pathloss = self.propagation.pathloss(dist)
                    else:
                        pathloss = mysplat.pathloss(t_x, t_y, sen.x, sen.y) + random.uniform(-0.25, 0.25)
                    f.write('{} {} {} {} {:.3f} {}\n'.format(t_x, t_y, sen.x, sen.y, power - pathloss, 1))



    def update_population(self, population_set, intruder, grid_len, min_dist, max_dist):
        '''Update the population (the TX candidate locations)
        Args:
            population_set -- set             -- the TX candidate locations
            intruder       -- tuple<int, int> -- the new selected TX
            grid_len       -- int             -- grid length
            min_dist       -- int             -- minimum distance between two TX. for far away sensors
            max_dist       -- int             -- maximum distance between two TX. for close by sensors
        '''
        if max_dist is None:
            cur_x = intruder[0]
            cur_y = intruder[1]
            for x in range(-min_dist, min_dist+1):
                for y in range(-min_dist, min_dist+1):
                    nxt_x = cur_x + x
                    nxt_y = cur_y + y
                    if 0 <= nxt_x < grid_len and 0 <= nxt_y < grid_len and Utility.distance(intruder, (nxt_x, nxt_y)) < min_dist:
                        if (nxt_x, nxt_y) in population_set:
                            population_set.remove((nxt_x, nxt_y))
        else:
            cur_x = intruder[0]
            cur_y = intruder[1]
            for x, y in population_set.copy():
                if not (min_dist <= Utility.distance(intruder, (x, y)) <= max_dist):
                    population_set.remove((x, y))



if __name__ == '__main__':

    # python generate.py -gd -rd data/matrix-train40 -sl 10 -rs 0 -nt 2 -mind 20
    # python generate.py -gd -rd data/matrix-train31 -sl 10 -cp 0.1 -rs 0 -nt 2
    # python generate.py -gd -rd data/matrix-test30 -sl 2 -cp 1 -rs 1 -nt 2

    # python generate.py -gd -rd data/61test -sl 2 rs 100 -nt 5 -ntup -mind 1

    parser = argparse.ArgumentParser(description='Localize multiple transmitters')

    parser.add_argument('-gs', '--generate_sensor', action='store_true')
    parser.add_argument('-gd', '--generate_data', action='store_true')
    parser.add_argument('-ipsn', '--ipsn', action='store_true')
    parser.add_argument('-spt', '--splat', action='store_true')
    parser.add_argument('-al', '--alpha', nargs=1, type=float, default=[Default.alpha], help='the slope of pathloss')
    parser.add_argument('-st', '--std', nargs=1, type=float, default=[Default.std], help='the standard deviation zero mean Guassian')
    parser.add_argument('-gl', '--grid_length', nargs=1, type=int, default=[Default.grid_length], help='the length of the grid')
    parser.add_argument('-cl', '--cell_length', nargs=1, type=float, default=[Default.cell_length], help='the length of a cell')
    parser.add_argument('-rs', '--random_seed', nargs=1, type=int, default=[Default.random_seed], help='random seed')
    parser.add_argument('-sd', '--sensor_density', nargs=1, type=int, default=[Default.sen_density], help='number of sensors')
    parser.add_argument('-po', '--power', nargs=1, type=float, default=[Default.power], help='the power of the transmitter')
    parser.add_argument('-cp', '--cell_percentage', nargs=1, type=float, default=[Default.cell_percentage], help='percentage of cells being labels')
    parser.add_argument('-sl', '--sample_per_label', nargs=1, type=int, default=[Default.sample_per_label], help='# of samples per label')
    parser.add_argument('-rd', '--root_dir', nargs=1, type=str, default=[Default.root_dir], help='the root directory for the images')
    parser.add_argument('-nl', '--noise_floor', nargs=1, type=str, default=[Default.noise_floor], help='RSSI cannot be lower than noise floor')
    parser.add_argument('-nt', '--num_tx', nargs=1, type=int, default=[Default.num_tx], help='number of transmitters')
    parser.add_argument('-mind', '--min_dist', nargs=1, type=int, default=[Default.min_dist], help='minimum distance between intruders')
    parser.add_argument('-maxd', '--max_dist', nargs=1, type=int, default=[None], help='maximum distance between intruders')
    parser.add_argument('-ntup', '--num_tx_upbound', action='store_true', help='if yes, then generate [1, ntx] number of TX')
    parser.add_argument('-ed', '--edge', nargs=1, type=int, default=[Default.edge], help='no TX at the edge')
    parser.add_argument('-vp', '--vary_power', nargs=1, type=float, default=[0], help='varying power amount')

    args = parser.parse_args()

    random_seed = args.random_seed[0]
    grid_length = args.grid_length[0]
    sensor_density = args.sensor_density[0]
    num_tx = args.num_tx[0]

    # python generate.py -gs -rs 0 -sd 100
    if args.generate_sensor:
        print('generating sensor')
        GenerateSensors.random(grid_length, sensor_density, random_seed, f'data/sensors/{grid_length}-{sensor_density}-{random_seed}')

    if args.generate_data:
        alpha       = args.alpha[0]
        std         = args.std[0]
        cell_length = args.cell_length[0]
        power       = args.power[0]
        cell_percentage  = args.cell_percentage[0]
        sample_per_label = args.sample_per_label[0]
        root_dir    = args.root_dir[0]
        noise_floor = args.noise_floor[0]
        min_dist    = args.min_dist[0]
        max_dist    = args.max_dist[0]
        num_tx_upbound = args.num_tx_upbound
        edge        = args.edge[0]
        vary_power  = args.vary_power[0]
        splat       = args.splat
        if splat:
            mysplat = Splat('data_splat/pl_map_array.json')

        print(f'generating {num_tx} TX data')

        gd = GenerateData(random_seed, alpha, std, grid_length, cell_length, sensor_density, noise_floor)
        gd.generate(power, cell_percentage, sample_per_label, f'data/sensors/{grid_length}-{sensor_density}-{random_seed}', root_dir,\
                    num_tx, num_tx_upbound, min_dist, max_dist, edge, vary_power, splat)

    if args.ipsn:  # only in training dataset
        # python generate.py -ipsn -rd data/100test -rs 100 -nt 5 -ntup -mind 1
        root_dir    = args.root_dir[0]
        power       = args.power[0]
        alpha       = args.alpha[0]
        std         = args.std[0]
        cell_length = args.cell_length[0]
        noise_floor = args.noise_floor[0]
        splat       = args.splat
        if splat:
            mysplat = Splat('data_splat/pl_map_array.json')
        root_dir += '-ipsn'
        gd = GenerateData(random_seed, alpha, std, grid_length, cell_length, sensor_density, noise_floor)
        gd.generate_ipsn(power, f'data/sensors/{grid_length}-{sensor_density}-{random_seed}', root_dir, splat)
        # the relationship between the deep learning root_dir and the ipsn root_dir is a difference of "-ipsn" suffix
